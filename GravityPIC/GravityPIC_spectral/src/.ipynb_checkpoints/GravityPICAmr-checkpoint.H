#ifndef _GRAVITYPICAMR_H_
#define _GRAVITYPICAMR_H_

#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>

#include "particles/ElectrostaticParticleContainer.H"

class GravityPICAmr
    : public amrex::AmrCore
{
public:

    GravityPICAmr ();
    virtual ~GravityPICAmr () = default;

    // AmrCore creates Geom(lev), DistributionMap(lev), boxArray(lev).
    // main.cpp will be updated to read from those accessors instead of
    // building/holding its own copies.

    // Initialize level 0 and particles
    void InitData();

    // Run the PIC loop
    void Evolve();

    // update masks after regrid
    void RebuildMasks();

    // --- fields
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > phi;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > rhs;
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> > eField;

    // build mask multifabs
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> masks;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> gather_masks;

    // --- particle container
    std::unique_ptr<ElectrostaticParticleContainer> myPC;

    // --- ErrorEst tags cells for regridding based on criterion in state_error
    virtual void ErrorEst (int lev, amrex::TagBoxArray& tags,
                            amrex::Real time, int ngrow) override;

    // make a new level (when is this used?)
    virtual void MakeNewLevelFromScratch (int lev, amrex::Real time,
                                           const amrex::BoxArray& ba,
                                           const amrex::DistributionMapping& dm) override;
    
    // make a new level from a coarser one
    virtual void MakeNewLevelFromCoarse (int lev, amrex::Real time,
                                          const amrex::BoxArray& ba,
                                          const amrex::DistributionMapping& dm) override;

    // change shape of a previously made level 
    virtual void RemakeLevel (int lev, amrex::Real time,
                               const amrex::BoxArray& ba,
                               const amrex::DistributionMapping& dm) override;

    virtual void ClearLevel (int lev) override;
 
private:
 
    // shared by MakeNewLevelFromScratch / MakeNewLevelFromCoarse / RemakeLevel --
    // reallocates phi, rhs, eField at level lev on the given ba/dm and zeroes them out. 
    // the data is recalculated every step from particle data so the values here do not matter, just the shape
    void AllocLevelData (int lev, const amrex::BoxArray& ba,
                          const amrex::DistributionMapping& dm);
 
    static constexpr int Ncomp = 1;

    // density threshold for tagging -- read from the inputs file as "rho_err"
    amrex::Real rho_err = 1.0;

    // define all global variables
    int n_buffer;
    int max_step;
    int boxsize;
    int bc;
    int plot_int = -1;
    int particle_output_int = -1;

    amrex::Real dt;
    std::string ic_file;
};

#endif 