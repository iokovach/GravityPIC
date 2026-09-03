#ifndef PARTICLE_PUSHER_H_
#define PARTICLE_PUSHER_H_

#include "field_solver/FieldSolver.H"
#include "field_solver/FieldSolver_K.H"

#include <AMReX_FillPatchUtil.H>
#include <AMReX_InterpBndryData.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_FFT_Poisson.H>

using namespace amrex;

namespace FieldSolver {

// Zeros out rho on cells right at the coarse/fine boundary before the multigrid solve
void fixRHSForSolve (const Vector<MultiFab*>& rhs,
                     const Vector<const iMultiFab*>& masks,
                     const Vector<Geometry>& geom, const IntVect& ratio) {
    int num_levels = rhs.size();
    for (int lev = 0; lev < num_levels; ++lev) {
        MultiFab& fine_rhs = *rhs[lev];
        const auto& mask = *masks[lev];
        const BoxArray& fine_ba = fine_rhs.boxArray();
        const DistributionMapping& fine_dm = fine_rhs.DistributionMap();
        MultiFab fine_bndry_data(fine_ba, fine_dm, 1, 1);
        zeroOutBoundary(fine_rhs, fine_bndry_data, mask);
    }
}

// Average the 2^D cells touching each node onto that node.
void average_cellcenter_to_node (MultiFab& nd, const MultiFab& cc)
{
    for (MFIter mfi(nd); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto nd_arr = nd[mfi].array();
        auto const cc_arr = cc[mfi].const_array();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real sum = 0.0;
            for (int kk = -1; kk <= 0; ++kk)
            for (int jj = -1; jj <= 0; ++jj)
            for (int ii = -1; ii <= 0; ++ii) {
                sum += cc_arr(i+ii, j+jj, k+kk);
            }
            nd_arr(i,j,k,0) = sum / Real(8.0); 
        });
    }
}

// Multi-level AMR Poisson solve
void computePhi (const Vector<const MultiFab*>& rhs,
                 const Vector<MultiFab*>& phi,
                 const Vector<BoxArray>& grids,
                 const Vector<DistributionMapping>& dm,
                 const Vector<Geometry>& geom,
                 const Vector<const iMultiFab*>& masks, int bc) {

    int num_levels = rhs.size();

    // Make a scratch copy of rhs so fixRHSForSolve can zero out boundary
    Vector<std::unique_ptr<MultiFab> > tmp_rhs(num_levels);
    for (int lev = 0; lev < num_levels; ++lev) {
        tmp_rhs[lev].reset(new MultiFab(rhs[lev]->boxArray(), dm[lev], 1, 0));
        MultiFab::Copy(*tmp_rhs[lev], *rhs[lev], 0, 0, 1, 0);
    }

    IntVect ratio(AMREX_D_DECL(2, 2, 2));
    // zero out fine boundary -- those values will be populated by the coarse values
    fixRHSForSolve(GetVecOfPtrs(tmp_rhs), masks, geom, ratio);

    int verbose = 2;
    // may need to toggle this or toggle number of iterations to reach tol
    Real rel_tol = 1.0e-8;
    Real abs_tol = 1.0e-10;

    Vector<Geometry>            level_geom(1);
    Vector<BoxArray>            level_grids(1);
    Vector<DistributionMapping> level_dm(1);
    Vector<MultiFab*>           level_phi(1);
    Vector<const MultiFab*>     level_rhs(1);

    for (int lev = 0; lev < num_levels; ++lev) {
        level_phi[0]   = phi[lev];
        level_rhs[0]   = tmp_rhs[lev].get();
        level_geom[0]  = geom[lev];
        level_grids[0] = grids[lev];
        level_dm[0]    = dm[lev];

        if (lev == 0) {
            // Poisson Method needs cell centered data 
            MultiFab rhs0_cc(level_grids[0], level_dm[0], 1, 0);
            MultiFab phi0_cc(level_grids[0], level_dm[0], 1, 1);
            
            amrex::average_node_to_cellcenter(rhs0_cc, 0, *level_rhs[0], 0, 1);
            
            if (bc != 0)
            {
                // calling Poisson Method with no explicit boundary descriptors assumes periodic in all directions
                amrex::FFT::Poisson<MultiFab> my_poisson(level_geom[0]);
                my_poisson.solve(phi0_cc, rhs0_cc);
    
                phi0_cc.FillBoundary(level_geom[0].periodicity());
                average_cellcenter_to_node(*phi[0], phi0_cc);
            }
            else 
            { 
                //Dirichlet zero BC: needs odd boundary descriptors
                Array<std::pair<amrex::FFT::Boundary,amrex::FFT::Boundary>,AMREX_SPACEDIM> boundary;
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    boundary[idim] = std::make_pair(amrex::FFT::Boundary::odd, amrex::FFT::Boundary::odd);
                }
                
                amrex::FFT::Poisson<MultiFab> poisson_solver(level_geom[0], boundary);
                poisson_solver.solve(phi0_cc, rhs0_cc);
                
                phi0_cc.FillBoundary(level_geom[0].periodicity());
                average_cellcenter_to_node(*phi[0], phi0_cc);
            }
            }
        else
            { // finite diff for higher levels
            // set up the discretized Laplacian operator on this level's grid
            MLNodeLaplacian linop(level_geom, level_grids, level_dm);
    
            // Dirichlet boundary conditions on every domain face
            linop.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                            LinOpBCType::Dirichlet,
                                            LinOpBCType::Dirichlet)},
                {AMREX_D_DECL(LinOpBCType::Dirichlet,
                              LinOpBCType::Dirichlet,
                              LinOpBCType::Dirichlet)});
    
            linop.setLevelBC(0, nullptr);
    
            // sigma is the coefficient in the more general operator
            // div(sigma * grad(phi)) = rhs
            // setting it to 1 everywhere recovers the plain Poisson equation grad^2 phi = rhs.
            MultiFab sigma(level_grids[0], level_dm[0], 1, 0);
            sigma.setVal(1.0);
            linop.setSigma(0, sigma);
    
            MLMG mlmg(linop);
            mlmg.setMaxIter(100);
            mlmg.setMaxFmgIter(0);
            mlmg.setVerbose(verbose);
            mlmg.setBottomVerbose(0);
    
            mlmg.solve(level_phi, level_rhs, rel_tol, abs_tol);
            }

        if (lev < num_levels-1) {

            PhysBCFunctNoOp cphysbc, fphysbc;

            int lo_bc[] = {AMREX_D_DECL(BCType::int_dir, BCType::int_dir, BCType::int_dir)};
            int hi_bc[] = {AMREX_D_DECL(BCType::int_dir, BCType::int_dir, BCType::int_dir)};

            Vector<BCRec> bcs(1, BCRec(lo_bc, hi_bc));
            NodeBilinear mapper;

            // Interpolate this (coarser) level's just-solved phi onto the
            // next-finer level, to seed/constrain its boundary values before
            // that level is solved in the next loop iteration.
            amrex::InterpFromCoarseLevel(*phi[lev+1], 0.0, *phi[lev],
                                         0, 0, 1, geom[lev], geom[lev+1],
                                         cphysbc, 0, fphysbc, 0,
                                         IntVect(AMREX_D_DECL(2, 2, 2)), &mapper, bcs, 0);
        }
    }

    for (int lev = 0; lev < num_levels; ++lev) {
        const Geometry& gm = geom[lev];
        phi[lev]->FillBoundary(gm.periodicity());
    }
}

// Averages down a fine level's data onto the corresponding coarse cells
void sumFineToCrseNodal (const MultiFab& fine, MultiFab& crse,
                         const Geometry& cgeom, const IntVect& ratio) {

    const BoxArray& fine_BA = fine.boxArray();
    const DistributionMapping& fine_dm = fine.DistributionMap();
    BoxArray coarsened_fine_BA = fine_BA;
    coarsened_fine_BA.coarsen(ratio);

    MultiFab coarsened_fine_data(coarsened_fine_BA, fine_dm, 1, 0);

    auto mask = OwnerMask(coarsened_fine_data, cgeom.periodicity());

    for (MFIter mfi(coarsened_fine_data); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto crse_data = coarsened_fine_data[mfi].array();
        const auto fine_data = fine[mfi].array();
    const auto mskfab = mask->const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                                   if (mskfab(i,j,k)) {
                                       sum_fine_to_crse_nodal(i, j, k, crse_data, fine_data, ratio);
                                   } else {
                                       crse_data(i,j,k) = Real(0.0);
                                   }
                               });
    }

    crse.ParallelCopy(coarsened_fine_data, cgeom.periodicity(), FabArrayBase::ADD);
}

// Zeros out input_data on cells flagged by mask (used to clear the RHS at
// coarse/fine boundaries in fixRHSForSolve above).
void zeroOutBoundary (MultiFab& input_data,
                      MultiFab& bndry_data,
                      const iMultiFab& mask) {
    bndry_data.setVal(0.0, 1);
    for (MFIter mfi(input_data); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto input_arr = input_data[mfi].array();
        auto bndry_arr = bndry_data[mfi].array();
        auto mask_arr = mask[mfi].array();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                                   zero_out_bndry(i, j, k, input_arr, bndry_arr, mask_arr);
                               });
    }
    bndry_data.FillBoundary();
}

// flag cells close to a coarse/fine boundary 
Vector<std::unique_ptr<iMultiFab> > getLevelMasks
                   (const Vector<BoxArray>& grids,
                    const Vector<DistributionMapping>& dmap,
                    const Vector<Geometry>& geom,
                    const int ncells) {

    int num_levels = grids.size();
    BL_ASSERT(num_levels == dmap.size());

    int covered = 0;
    int notcovered = 1;
    int physbnd = 1;
    int interior = 0;

    amrex::Vector<std::unique_ptr<amrex::iMultiFab> > masks(num_levels);

    for (int lev = 0; lev < num_levels; ++lev) {
        BoxArray nba = grids[lev];
        nba.surroundingNodes();

        // tmp_mask has 1 in uncovered ghost cells or cells outside the physical boundary.
        FabArray<BaseFab<int> > tmp_mask(nba, dmap[lev], 1, ncells);
        tmp_mask.BuildMask(geom[lev].Domain(), geom[lev].periodicity(),
                           covered, notcovered, physbnd, interior);
        masks[lev].reset(new iMultiFab(nba, dmap[lev], 1, 0));

        for (MFIter mfi(tmp_mask); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            const auto tmp_arr = tmp_mask[mfi].array();
            auto mask_arr = (*masks[lev])[mfi].array();
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                                       build_mask(i, j, k, tmp_arr, mask_arr, ncells);
                                   });
        }
    }

    return masks;
}

// Computes the field E = -grad(phi) at mesh nodes from the solved potential
//(the actual gradient stencil is in compute_E_nodal, in FieldSolver_K.H)
void computeE (const Vector<std::array<MultiFab*, AMREX_SPACEDIM> >& E,
               const Vector<const MultiFab*>& phi,
               const Vector<Geometry>& geom) {

    const int num_levels = E.size();

    for (int lev = 0; lev < num_levels; ++lev) {
        const auto& gm = geom[lev];
        const auto dx = gm.CellSizeArray();
        for (MFIter mfi(*phi[lev]); mfi.isValid(); ++mfi) {
            Box bx = mfi.validbox();
            bx.grow(1);
            auto Ex_arr = (*E[lev][0])[mfi].array();
            auto Ey_arr = (*E[lev][1])[mfi].array();
            // FIX: added the z-component array -- the original only ever
            // computed/filled Ex and Ey, silently leaving Ez untouched.
            auto Ez_arr = (*E[lev][2])[mfi].array();

            const auto phi_arr = (*phi[lev])[mfi].array();
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                                       compute_E_nodal(i, j, k,
                                                       Ex_arr, Ey_arr, Ez_arr,
                                                       phi_arr, dx);
                                   });
        }

        // what does this do?
        E[lev][0]->FillBoundary(gm.periodicity());
        E[lev][1]->FillBoundary(gm.periodicity());
        E[lev][2]->FillBoundary(gm.periodicity());
    }
}

};

#endif