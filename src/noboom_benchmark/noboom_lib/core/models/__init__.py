"""Local benchmark model implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HPAD": (".hpad", "HPAD"),
    "HPADAnomalyDetector": (".hpad", "HPADAnomalyDetector"),
    "HPADLoss": (".hpad", "HPADLoss"),
    "IGAD": (".igad", "IGAD"),
    "IGADAnomalyDetector": (".igad", "IGADAnomalyDetector"),
    "IGADLoss": (".igad", "IGADLoss"),
    "KANAD": (".kan_ad", "KANAD"),
    "KANADAnomalyDetector": (".kan_ad", "KANADAnomalyDetector"),
    "KANADModel": (".kan_ad", "KANADModel"),
    "PhysDiff": (".physdiff", "PhysDiff"),
    "PhysDiffAnomalyDetector": (".physdiff", "PhysDiffAnomalyDetector"),
    "PhysDiffLoss": (".physdiff", "PhysDiffLoss"),
    "AnomalyTransformer": (".anomaly_transformer", "AnomalyTransformer"),
    "AnomalyTransformerAnomalyDetector": (".anomaly_transformer", "AnomalyTransformerAnomalyDetector"),
    "AnomalyTransformerLoss": (".anomaly_transformer", "AnomalyTransformerLoss"),
    "ALoRa": (".alora", "ALoRa"),
    "ALoRaAnomalyDetector": (".alora", "ALoRaAnomalyDetector"),
    "ALoRaLoss": (".alora", "ALoRaLoss"),
    "DCdetector": (".dcdetector", "DCdetector"),
    "DCdetectorAnomalyDetector": (".dcdetector", "DCdetectorAnomalyDetector"),
    "CATCH": (".catch", "CATCH"),
    "CATCHAnomalyDetector": (".catch", "CATCHAnomalyDetector"),
    "CATCHLoss": (".catch", "CATCHLoss"),
    "DADA": (".dada", "DADA"),
    "DADAAnomalyDetector": (".dada", "DADAAnomalyDetector"),
    "RTdetector": (".rtdetector", "RTdetector"),
    "RTdetectorAnomalyDetector": (".rtdetector", "RTdetectorAnomalyDetector"),
    "RTdetectorLoss": (".rtdetector", "RTdetectorLoss"),
    "CAROTS": (".carots", "CAROTS"),
    "CAROTSAnomalyDetector": (".carots", "CAROTSAnomalyDetector"),
    "CAROTSLoss": (".carots", "CAROTSLoss"),
    "PaAno": (".paano", "PaAno"),
    "PaAnoAnomalyDetector": (".paano", "PaAnoAnomalyDetector"),
    "OracleAD": (".oraclead", "OracleAD"),
    "OracleADAnomalyDetector": (".oraclead", "OracleADAnomalyDetector"),
    "OracleADLoss": (".oraclead", "OracleADLoss"),
    "ScatterAD": (".scatterad", "ScatterAD"),
    "ScatterADAdam": (".scatterad", "ScatterADAdam"),
    "ScatterADAnomalyDetector": (".scatterad", "ScatterADAnomalyDetector"),
    "ScatterADLoss": (".scatterad", "ScatterADLoss"),
    "ITransformer": (".catch_paper_neural", "ITransformer"),
    "ModernTCN": (".catch_paper_neural", "ModernTCN"),
    "DualTF": (".catch_paper_neural", "DualTF"),
    "PatchTST": (".catch_paper_neural", "PatchTST"),
    "DLinear": (".catch_paper_neural", "DLinear"),
    "NLinear": (".catch_paper_neural", "NLinear"),
    "TFAD": (".catch_paper_neural", "TFAD"),
    "AutoEncoder": (".catch_paper_neural", "AutoEncoder"),
    "NeuralReconstructionAnomalyDetector": (
        ".catch_paper_neural",
        "NeuralReconstructionAnomalyDetector",
    ),
    "NeuralReconstructionLoss": (".catch_paper_neural", "NeuralReconstructionLoss"),
    "ReconstructionAnomalyDetector": (".catch_paper_neural", "ReconstructionAnomalyDetector"),
    "OCSVMAD": (".detectors", "OCSVMAD"),
    "PCAAD": (".detectors", "PCAAD"),
    "HBOSAD": (".detectors", "HBOSAD"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
