import FWCore.ParameterSet.Config as cms

process = cms.Process("SUPERCLS")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )

process.source = cms.Source("EmptySource")
process.dummySource = cms.ESSource("EmptyESSource", 
      recordName=cms.string("TfGraphRecord"), firstValid=cms.vuint32(1), iovIsRunNotTime = cms.bool(True))

process.DummyTracksters = cms.EDProducer("DummyTracksterProducer")


#from RecoHGCal.TICL.superclusteringProducer_cfi import superclusteringProducer as _superclusteringProducer
process.load("RecoHGCal.TICL.superclusteringTf_cff")
process.superclusteringTf.FileName = "RecoHGCal/TICL/data/tf_models/supercls_v2.pb"

process.superclusteringProducer = cms.EDProducer('SuperclusteringProducer',
  tfDnnLabel = cms.string('superclusteringTf'),
  dnnVersion = cms.string('alessandro-v2'),
  trackstersclue3d = cms.InputTag('DummyTracksters'),
  nnWorkingPoint = cms.double(0.51),
  mightGet = cms.optional.untracked.vstring
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('dummyTrackstersSupercls.root')
    ,outputCommands = cms.untracked.vstring("keep *")

)

process.p = cms.Path(process.DummyTracksters*process.superclusteringProducer)

process.e = cms.EndPath(process.out)

process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.debugModules = ["superclusteringProducer"]