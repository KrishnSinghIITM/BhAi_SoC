# End-to-End Dual-Engine Evaluation Report (All 9 Defect Classes)

**Timestamp**: 2026-10-18 10:52:57 UTC
**Tier 1 Model**: YOLO11-Nano INT8 (2.92 MB, 11.8 ms inference)
**Tier 2 Model**: Llama 3.2 1B-Instruct ExecuTorch 4-Bit (8da4w, 24.89 MB activation RAM)

| Defect Code | Defect Name | YOLO Conf | Generation Time | Tok / Sec | IPC-A-610 On-Device Advisory |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **SH** | Short Circuit | 94.0% | 3.23s | **20.4** | - Severity & IPC Class: 3.1 - Component: 2A01 (Airplane) - Location: 0010 - Type: Aircraft Wing  Relevant Documentation: * IPC-A-6... |
| **HB** | Hole Break | 98.0% | 2.9s | **21.8** | - Severity & IPC Class: 3 - Defect Type: Hole Break (HB) - IPC Class 1: PTH  ## Step 1: Review the standard for PCB inspection and... |
| **BMFO** | Board Mask Flaw | 88.0% | 2.86s | **21.7** | - Severity & IPC Class: 1.2 - Rework Timeframe: 6 weeks  **Engineering Notes**  * The board mask flaw (BMFO) is detected on the ou... |
| **OP** | Open Circuit | 91.0% | 2.95s | **22.1** | - Severity & IPC Class: 1.2, 4 - Defect Type: Open Circuit (OP) - IPC Class: A11  ## Step 1: Review the standard PCB inspection pr... |
| **SC** | Scratch | 85.0% | 2.95s | **22.7** | - Severity & IPC Class: 1 - Defect Type: Scratch (SC) - IPC Class: 2A10B - Surface Layer: Copper / Mask  ## Step 1: Determine the ... |
| **SP** | Spur | 82.0% | 2.99s | **22.8** | - Severity & IPC Class: 3 - Defect Type: Spur (SP) - IPC Class: 1A  Recommendation: The spurs on the printed circuit boards need t... |
| **MB** | Mouse Bite | 79.0% | 2.94s | **22.8** | - Severity & IPC Class: 3.1 - Defect Type: Mouse Bite (MB) - IPC Class: 2A10  Recommendations: - Remove the affected component fro... |
| **CS** | Conductor Spacing | 86.0% | 2.83s | **22.3** | - Severity & IPC Class: 1A - Defect Type: Conductor Spacing (CS) - IPC Class: 2B  ## Step 1: Review the standard PCB inspection pr... |
| **CFO** | Copper-Free Error | 78.0% | 2.93s | **22.2** | - Severity & IPC Class: 3 - Defect Type: CFO (Copper-Free Error) - IPC Class 1 and 2:     - Severity Level: Medium  Engineering Re... |
