#include "GravityPICAmr.H"
#include "field_solver/FieldSolver.H"
#include "particles/ElectrostaticParticleContainer.H"
#include "diagnostics/FieldIO.H"

// constructor -- make an AmrCore object and read in simulation parameters
GravityPICAmr::GravityPICAmr()
    : amrex::AmrCore()
{
    amrex::ParmParse pp;
    amrex::Array<int, AMREX_SPACEDIM> pdc;
    
    pp.query("rho_err", rho_err);
    pp.get("n_buffer", n_buffer);
    pp.get("boxsize", boxsize);
    pp.get("max_step", max_step);
    pp.get("dt", dt);
    pp.get("ic_file", ic_file);
    pp.get("geometry.is_periodic", pdc);

    pp.query("plot_int", plot_int);
    pp.query("particle_output_int", particle_output_int);

    bc = pdc[0];
}

// function to build boxarrays and particles
void
GravityPICAmr::InitData()
{
    // Build the initial BoxArray, Geom and DM at level 0 
    InitFromScratch(0.0);

    // mask out coarse and fine regions for field gathering later
    RebuildMasks();

    // Construct particle container using the objects we just initialized
    myPC = std::make_unique<ElectrostaticParticleContainer>(
        Geom(),
        DistributionMap(),
        boxArray(),
        refRatio()
    );

    // read particle data and put them in the container
    myPC->InitParticles(ic_file);
}

void GravityPICAmr::RebuildMasks()
{
    const int nlev = finestLevel() + 1;

    // Get the current AMR hierarchy from AmrCore
    amrex::Vector<amrex::BoxArray> grids(nlev);
    amrex::Vector<amrex::DistributionMapping> dm(nlev);
    amrex::Vector<amrex::Geometry> geom(nlev);

    for (int lev = 0; lev < nlev; ++lev) {
        grids[lev] = boxArray(lev);
        dm[lev]    = DistributionMap(lev);
        geom[lev]  = Geom(lev);
    }

    // Rebuild the two sets of masks
    masks = FieldSolver::getLevelMasks(grids, dm, geom);

    gather_masks = FieldSolver::getLevelMasks(
        grids, dm, geom, n_buffer + 1
    );
}

void
GravityPICAmr::Evolve()
{
    // main PIC loop
    for (int step = 0; step <= max_step; ++step) {

        myPC->DepositCharge(GetVecOfPtrs(rhs));
        
        // subtract off the mean to enforce -- use sum_uniqe since this is nodal
        // Domain gives us the region as a cell-centered box 
        // so it does not overcount nodes lying between mpi ranks or at periodic bdy
        amrex::Real mean_rho = rhs[0]->sum_unique(0, false, geom[0].periodicity()) / geom[0].Domain().numPts();  
        
        rhs[0]->plus(-mean_rho, 0, 1);

        FieldSolver::computePhi(GetVecOfConstPtrs(rhs), GetVecOfPtrs(phi),
                                boxArray(), DistributionMap(), Geom(),
                                GetVecOfConstPtrs(masks), bc);

        FieldSolver::computeE(GetVecOfArrOfPtrs(eField), GetVecOfConstPtrs(phi), Geom());

        myPC->FieldGather(GetVecOfArrOfConstPtrs(eField),
                         GetVecOfConstPtrs(gather_masks));

        if ((particle_output_int > 0) && (step % particle_output_int == 0)) {
            myPC->writeParticles(step);
        }

        if ((plot_int > 0) && (step % plot_int == 0)) {
            WritePlotFile(GetVecOfConstPtrs(rhs), GetVecOfConstPtrs(phi), GetVecOfArrOfConstPtrs(eField), *myPC, geom, step);
        }

        myPC->Evolve(GetVecOfArrOfConstPtrs(eField), GetVecOfConstPtrs(rhs), dt);
        
        // periodic BCs are default in Redistribute
        myPC->Redistribute();
    }
}


void
GravityPICAmr::ErrorEst (int lev, amrex::TagBoxArray& tags,
                          amrex::Real /*time*/, int /*ngrow*/)
{
    amrex::Print() << "ErrorEst called: lev = " << lev
                   << ", time = " << time << "\n";

    const amrex::MultiFab& rho = *rhs[lev];
 
    for (amrex::MFIter mfi(rho, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto& rho_arr = rho.const_array(mfi);
        auto tag_arr = tags.array(mfi);
        const amrex::Real threshold = rho_err;
 
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // using the node at cell's low corner as a stand-in for the whole cell's density
            // averaging the surrounding nodes would be more accurate
            if (std::abs(rho_arr(i,j,k)) > threshold) {
                tag_arr(i,j,k) = amrex::TagBox::SET;
            }
        });
    }
}

void
GravityPICAmr::AllocLevelData (int lev, const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm)
{
    // Function to allocate a box array of correct regridded size/dimension
    amrex::BoxArray nba = ba;
    nba.surroundingNodes();

    rhs[lev] = std::make_unique<amrex::MultiFab>(nba, dm, Ncomp, 1);
    phi[lev] = std::make_unique<amrex::MultiFab>(nba, dm, Ncomp, 2);
    rhs[lev]->setVal(0.0);
    phi[lev]->setVal(0.0);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        eField[lev][idim] = std::make_unique<amrex::MultiFab>(nba, dm, Ncomp, 1);
        eField[lev][idim]->setVal(0.0);
    }
}

// make a new level where there was none
void
GravityPICAmr::MakeNewLevelFromScratch (int lev, amrex::Real /*time*/,
                                         const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm)
{
    if (static_cast<int>(rhs.size()) <= lev) {
        rhs.resize(lev+1);
        phi.resize(lev+1);
        eField.resize(lev+1);
    }
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);
    AllocLevelData(lev, ba, dm);
    
    RebuildMasks();
}

// make a new level where there was none -- this is the same as from scratch because 
// rhs/phi/eField are fully recomputed from deposited charge every step
void
GravityPICAmr::MakeNewLevelFromCoarse (int lev, amrex::Real /*time*/,
                                        const amrex::BoxArray& ba,
                                        const amrex::DistributionMapping& dm)
{

    if (static_cast<int>(rhs.size()) <= lev) {
        rhs.resize(lev+1);
        phi.resize(lev+1);
        eField.resize(lev+1);
    }
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);
    AllocLevelData(lev, ba, dm);

    RebuildMasks();
}

// change the shape of a n existing level's BoxArray/DistributionMapping 
void
GravityPICAmr::RemakeLevel (int lev, amrex::Real /*time*/,
                             const amrex::BoxArray& ba,
                             const amrex::DistributionMapping& dm)
{

    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);
    AllocLevelData(lev, ba, dm);

    RebuildMasks();
}

void
GravityPICAmr::ClearLevel (int lev)
{
    rhs[lev].reset();
    phi[lev].reset();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        eField[lev][idim].reset();
    }
}

