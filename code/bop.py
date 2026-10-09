"""Thin driver for BOPTEST emulator FMUs (IBPSA project1-boptest, commit 0f8a467) via fmpy co-simulation, without the web service.
Requires LD_LIBRARY_PATH to include ./fmu_libs/lib (libgfortran.so.4, conda-forge libgfortran4)."""
import os, numpy as np
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
HERE = os.path.dirname(os.path.abspath(__file__))

class Emulator:
    def __init__(self, case='bestest_hydronic_heat_pump', name='a'):
        fmu = os.path.join(HERE, 'boptest', 'testcases', case, 'models', 'wrapped.fmu')
        self.md = read_model_description(fmu, validate=False); self.dir = extract(fmu)
        self.vr = {v.name: v.valueReference for v in self.md.modelVariables}
        self.s = FMU2Slave(guid=self.md.guid, unzipDirectory=self.dir, modelIdentifier=self.md.coSimulation.modelIdentifier, instanceName=name)
        self.bases = [n[:-9] for n in self.vr if n.endswith('_activate')]   # every overwritable input of the test case
        self.s.instantiate()
    def init(self, t0):
        self.s.setupExperiment(startTime=t0); self.s.enterInitializationMode(); self.s.exitInitializationMode(); self.t = t0
    def step(self, dt, overrides=None):
        """overrides: dict input-base-name -> value (e.g. {'oveHeaPumY': 0.5}); non-listed inputs revert to the embedded controller."""
        for base in self.bases:
            on = overrides is not None and base in overrides
            self.s.setBoolean([self.vr[base + '_activate']], [bool(on)])          # the activate inputs are Boolean (fixed 2026-10-08: setReal was silently unreliable)
            if on: self.s.setReal([self.vr[base + '_u']], [float(overrides[base])])
        self.s.doStep(currentCommunicationPoint=self.t, communicationStepSize=dt); self.t += dt
    def get(self, names): return self.s.getReal([self.vr[n] for n in names])
    def state(self): return self.s.getFMUstate()
    def restore(self, st): self.s.setFMUstate(st)
    def close(self): self.s.terminate(); self.s.freeInstance()
