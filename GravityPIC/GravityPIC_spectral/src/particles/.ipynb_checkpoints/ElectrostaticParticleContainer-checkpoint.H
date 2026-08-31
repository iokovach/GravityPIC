#ifndef _ELECTROSTATICPARTICLECONTAINER_H_
#define _ELECTROSTATICPARTICLECONTAINER_H_

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrParGDB.H>

#include "PhysConst.H"

struct PIdx
{
    enum { // Particle Attributes stored in amrex::ParticleContainer's struct-of-arrays
          w = 0,
          vx, vy, vz,
          Ex, Ey, Ez,
          nattribs
    };
};

class MyParIter
    : public amrex::ParIter<0, 0, PIdx::nattribs>
{
public:
    using amrex::ParIter<0, 0, PIdx::nattribs>::ParIter;
    using RealVector = amrex::ParIter<0, 0, PIdx::nattribs>::ContainerType::RealVector;

    const std::array<RealVector, PIdx::nattribs>& GetAttribs () const {
        return GetStructOfArrays().GetRealData();
    }

    std::array<RealVector, PIdx::nattribs>& GetAttribs () {
        return GetStructOfArrays().GetRealData();
    }

    const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }

};

class ElectrostaticParticleContainer
    : public amrex::ParticleContainer<0, 0, PIdx::nattribs>
{
public:
    // build the particle container from an amr core object dynamically linked
    // to a ParGDB which reads the live state of AMR core -- how many levels etc
    ElectrostaticParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainer<0, 0, PIdx::nattribs> (amr_core->GetParGDB())
        {
            amrex::ParmParse pp("particles");
            mass = 1.0;  // default
            pp.query("mass", mass); 
        }

    void InitParticles (const std::string& filename);

    void pushX (const amrex::Real& dt);

    void DepositCharge (const amrex::Vector<amrex::MultiFab*>& rho);

    void FieldGather (const amrex::Vector<std::array<const amrex::MultiFab*, AMREX_SPACEDIM> >& E,
                      const amrex::Vector<const amrex::iMultiFab*>& masks);

    void Evolve (const amrex::Vector<std::array<const amrex::MultiFab*, AMREX_SPACEDIM> >& E,
                 const amrex::Vector<const amrex::MultiFab*>& rho, const amrex::Real& dt);

    void writeParticles (int n);

private:

    amrex::Real charge, mass;
};

#endif // _ELECTROSTATICPARTICLECONTAINER_H_
