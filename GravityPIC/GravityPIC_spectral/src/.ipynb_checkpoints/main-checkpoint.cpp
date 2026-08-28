#include "diagnostics/FieldIO.H"
#include "field_solver/FieldSolver.H"
#include "particles/ElectrostaticParticleContainer.H"

#include <AMReX.H>

#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <GravityPICAmr.H>

using namespace amrex;

void run_espic ();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    run_espic();

    amrex::Finalize();
}

void run_espic ()
{
    // timer for profiling
    BL_PROFILE("main()");
    
    // wallclock time
    const auto strt_total = amrex::second();
    
    // constructor - reads in parameters from inputs file
    //             - sizes multilevel arrays and data structures
    GravityPICAmr GravityPICAmr;
    
    // initialize AMR data
    GravityPICAmr.InitData();
    
    // advance solution to final time
    GravityPICAmr.Evolve();
    
    // wallclock time
    auto end_total = amrex::second() - strt_total;
    
    amrex::Print() << "\nTotal Time: " << end_total << '\n';

}
