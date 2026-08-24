#include "field_solver/FieldSolver.H"

#include "particles/ElectrostaticParticleContainer.H"
#include "particles/pusher/ParticlePusher_K.H"
#include "particles/field_gather/FieldGather_K.H"
#include "particles/deposition/ChargeDeposition_K.H"

#include <iomanip>

using namespace amrex;

// initialize particle structures
void ElectrostaticParticleContainer::InitParticles (int n_part) {

    // Only rank 0 (the "IO processor") creates the initial particle(s)
    // one MPI process runs this code, every process still needs
    // to eventually know about the particles -- that's what Redistribute()
    // at the bottom does: it moves/broadcasts particles to whichever
    // process actually owns the grid box the particle lives in.
    if ( ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber() ) {
        
        std::mt19937 gen(12345);

        //get box bounds to initialize particles within box
        const auto& geom = m_gdb->Geom(0);
        const auto plo = geom.ProbLoArray();
        const auto phi = geom.ProbHiArray();

        std::uniform_real_distribution<ParticleReal> dist_x(plo[0], phi[0]);
        std::uniform_real_distribution<ParticleReal> dist_y(plo[1], phi[1]);
        std::uniform_real_distribution<ParticleReal> dist_z(plo[2], phi[2]);
        
        for (int i = 0; i < n_part; ++i) {
            ParticleType p;
    
            // Every particle gets a globally unique ID and records which CPU/rank made it
            p.id()   = ParticleType::NextID();
            p.cpu()  = ParallelDescriptor::MyProc();
    
            // Set the particle's initial physical position (x, y, z).
            p.pos(0) = dist_x(gen);
            p.pos(1) = dist_y(gen);
            p.pos(2) = dist_z(gen);
    
            // "attribs" holds the particle's non-positional data: weight, velocity, and field values at the particle's location
            // PIdx is an enumeration -- gives names to a sequence of numbers 
            // note that velocities are stored as attributes and positions live in the particletype structure
            // this has something to do with optimizing the code for massively parallel runs....idk
            std::array<ParticleReal,PIdx::nattribs> attribs;
            attribs[PIdx::w]  = 1.0;
            attribs[PIdx::vx] = 0.0;
            attribs[PIdx::vy] = 0.0;
            attribs[PIdx::vz] = 0.0;
            attribs[PIdx::Ex] = 0.0;
            attribs[PIdx::Ey] = 0.0;
            attribs[PIdx::Ez] = 0.0;
    
            // Add to level 0, grid 0, and tile 0
            std::pair<int,int> key {0,0};
            auto& particle_tile = GetParticles(0)[key];
    
            particle_tile.push_back(p);
            particle_tile.push_back_real(attribs);
        }

    }
    // Redistribute() moves each particle to whichever 
    // MPI rank and grid box it should be in based on position
    Redistribute();
}

// call deposit_cic correctly for AMR
void
ElectrostaticParticleContainer::DepositCharge (const Vector<MultiFab*>& rho) {

    int num_levels = rho.size();
    int finest_level = num_levels - 1;

    // Each AMR level deposits its own particles onto its own rho MultiFab
    const int ng = rho[0]->nGrow();
    for (int lev = 0; lev < num_levels; ++lev) {
        rho[lev]->setVal(0.0, ng);
        const auto& gm = m_gdb->Geom(lev);
        auto plo = gm.ProbLoArray();
        auto dxi = gm.InvCellSizeArray();
        // MyParIter loops over all the particle "tiles" (chunks of particles grouped by which grid box they live in) 
        for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
            const Long np  = pti.numParticles();
            const auto& wp = pti.GetAttribs(PIdx::w);
            const auto& particles = pti.GetArrayOfStructs();

            amrex::Real m = this->mass;
            auto rhoarr = (*rho[lev])[pti].array();
            const auto wp_ptr = wp.data();
            const auto p_ptr = particles().data();
            // ParallelFor launches one "thread" per particle (on GPU) or loops serially (on CPU) 
            // each iteration deposits one particle's mass onto the surrounding mesh nodes of rhoarr
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                          deposit_cic(p_ptr[i], wp_ptr[i], m, rhoarr, plo, dxi);
                                   });
        }

        // handle deposition for particles on the boundary between boxes owned by different ranks
        rho[lev]->SumBoundary(0, 1, IntVect(1), gm.periodicity());
    }

    // For AMR runs with multiple levels, the fine level's mass density also
    // needs to be averaged down onto the coarse level's density for the coarse level Poisson solve
    for (int lev = finest_level - 1; lev >= 0; --lev) {
        FieldSolver::sumFineToCrseNodal(*rho[lev+1], *rho[lev], m_gdb->Geom(lev), m_gdb->refRatio(lev));
    }

    // Rescale the deposited mass density by 4piG 
    // (grad^2 phi = 4piG*rho 
    for (int lev = 0; lev < num_levels; ++lev) {
        rho[lev]->mult(PhysConst::FourPiG, ng);
    }
}

// call interpolate_cic correctly for AMR
void
ElectrostaticParticleContainer::
FieldGather (const Vector<std::array<const MultiFab*, AMREX_SPACEDIM> >& E,
             const Vector<const iMultiFab*>& masks) {

    const int num_levels = E.size();
    const int ng = E[0][0]->nGrow();

    if (num_levels == 1) {
        // Single-level case: no coarse/fine boundary handling needed,
        // so we can call the plain interpolate_cic directly
        const int lev = 0;
        const auto& gm = m_gdb->Geom(lev);
        auto plo = gm.ProbLoArray();
        auto dxi = gm.InvCellSizeArray();
        AMREX_ASSERT(OnSameGrids(lev, *E[lev][0]));

        for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
            auto& particles = pti.GetArrayOfStructs();
            auto p_ptr = particles().data();
            const Long np  = pti.numParticles();

            auto& attribs = pti.GetAttribs();
            auto Ex_p = attribs[PIdx::Ex].data();
            auto Ey_p = attribs[PIdx::Ey].data();
            auto Ez_p = attribs[PIdx::Ez].data();

            const auto& exarr = (*E[lev][0])[pti].array();
            const auto& eyarr = (*E[lev][1])[pti].array();
            const auto& ezarr = (*E[lev][2])[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                                       interpolate_cic(p_ptr[i], Ex_p[i], Ey_p[i], Ez_p[i],
                                                       exarr, eyarr, ezarr, plo, dxi);
                                   });
        }
        return;
    }

    // Multi-level (AMR) case below: build a coarse copy of the fine-level field so 
    // particles near the coarse/fine boundary can fall back to it
    const BoxArray& fine_BA = E[1][0]->boxArray();
    const DistributionMapping& fine_dm = E[1][0]->DistributionMap();
    BoxArray coarsened_fine_BA = fine_BA;
    
    coarsened_fine_BA.coarsen(IntVect(AMREX_D_DECL(2,2,2)));

    MultiFab coarse_Ex(coarsened_fine_BA, fine_dm, 1, 1);
    MultiFab coarse_Ey(coarsened_fine_BA, fine_dm, 1, 1);
    MultiFab coarse_Ez(coarsened_fine_BA, fine_dm, 1, 1);

    coarse_Ex.ParallelCopy(*E[0][0], 0, 0, 1, 1, 1);
    coarse_Ey.ParallelCopy(*E[0][1], 0, 0, 1, 1, 1);
    coarse_Ez.ParallelCopy(*E[0][2], 0, 0, 1, 1, 1);

    for (int lev = 0; lev < num_levels; ++lev) {
        const auto& gm = Geom(lev);

        BL_ASSERT(OnSameGrids(lev, *E[lev][0]));

        for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
            const auto& particles = pti.GetArrayOfStructs();
            const Long np  = pti.numParticles();

            auto& attribs = pti.GetAttribs();
            auto Ex_p = attribs[PIdx::Ex].data();
            auto Ey_p = attribs[PIdx::Ey].data();
            auto Ez_p = attribs[PIdx::Ez].data();

            const auto exarr = (*E[lev][0])[pti].array();
            const auto eyarr = (*E[lev][1])[pti].array();
            const auto ezarr = (*E[lev][2])[pti].array();

            auto p_ptr = particles().data();
            auto ploarr = gm.ProbLoArray();
            auto dxi = gm.InvCellSizeArray();

            if (lev == 0) {
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                                           interpolate_cic(p_ptr[i], Ex_p[i], Ey_p[i], Ez_p[i],
                                                           exarr, eyarr, ezarr, ploarr, dxi);
                                       });
            } else {
                // Finer levels use interpolate_cic_two_levels
                const auto& cgm = Geom(lev-1);
                auto cdxi = cgm.InvCellSizeArray();

                const auto cexarr = coarse_Ex[pti].array();
                const auto ceyarr = coarse_Ey[pti].array();
                const auto cezarr = coarse_Ez[pti].array();

                const auto maskarr = (*masks[lev])[pti].array();

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    interpolate_cic_two_levels(p_ptr[i], Ex_p[i], Ey_p[i], Ez_p[i],
                                               exarr, eyarr, ezarr,
                                               cexarr, ceyarr, cezarr, maskarr,
                                               ploarr, dxi, cdxi, lev);
                                       });
            }
        }
    }
}


// velocity update 
void ElectrostaticParticleContainer
::Evolve (const Vector<std::array<const MultiFab*, AMREX_SPACEDIM> >& E,
          const Vector<const MultiFab*>& rho, const Real& dt) {

    const int num_levels = E.size();

    for (int lev = 0; lev < num_levels; ++lev) {

        const auto& gm = m_gdb->Geom(lev);
        const RealBox& prob_domain = gm.ProbDomain();

        BL_ASSERT(OnSameGrids(lev, *rho[lev]));
        for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {

            auto& particles = pti.GetArrayOfStructs();
            const Long np  = pti.numParticles();

            auto& attribs = pti.GetAttribs();
            auto vxp = attribs[PIdx::vx].data();
            auto vyp = attribs[PIdx::vy].data();
            auto vzp = attribs[PIdx::vz].data();

            auto Exp = attribs[PIdx::Ex].data();
            auto Eyp = attribs[PIdx::Ey].data();
            auto Ezp = attribs[PIdx::Ez].data();

            auto p_ptr = particles().data();
            const auto plo = gm.ProbLoArray();
            const auto phi = gm.ProbHiArray();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                                       push_leapfrog(p_ptr[i].pos(0), p_ptr[i].pos(1), p_ptr[i].pos(2),
                                                     vxp[i], vyp[i], vzp[i],
                                                     Exp[i], Eyp[i], Ezp[i],
                                                     dt, plo, phi);
                                   });
        }
    }
}

// position update
void ElectrostaticParticleContainer::pushX (const Real& dt) {

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto& gm = m_gdb->Geom(lev);
        const RealBox& prob_domain = gm.ProbDomain();
        for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
            auto& particles = pti.GetArrayOfStructs();
            const Long np  = pti.numParticles();

            auto& attribs = pti.GetAttribs();
            auto vxp = attribs[PIdx::vx].data();
            auto vyp = attribs[PIdx::vy].data();
            auto vzp = attribs[PIdx::vz].data();

            auto p_ptr = particles().data();
            auto plo = gm.ProbLoArray();
            auto phi = gm.ProbHiArray();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                                       push_leapfrog_positions(p_ptr[i].pos(0), p_ptr[i].pos(1), p_ptr[i].pos(2),
                                                               vxp[i], vyp[i], vzp[i], dt, plo, phi);
                                   });

        }
    }
}