# **多模態情感分析實驗**

## **簡介**
本實驗實現了一個多模態情感分析模型，通過結合文本和圖像的特徵來預測情感（`positive`, `neutral`, `negative`）。模型使用了 BERT 作為文本特徵提取器，ResNet18 作為圖像特徵提取器，最終通過全連接層進行融合並完成分類任務。

---

## **文件結構**

```
.
├── train.py                # 主程序，包含數據加載、模型訓練和測試代碼
├── data/                   # 數據目錄，包含訓練、驗證和測試數據
│   ├── train.txt           # 訓練數據
│   ├── val.txt             # 驗證數據
│   ├── test_without_label_cleaned.txt  # 測試數據（無標籤）
│   ├── <guid>.txt          # 每個樣本對應的文本文件
│   ├── <guid>.jpg          # 每個樣本對應的圖像文件
├── requirements.txt        # 依賴包列表
└── README.md               # 項目說明文件
```

---

## **環境要求**

運行代碼所需的環境如下：
- Python 3.8+
- PyTorch 2.0.0
- torchvision 0.15.0
- transformers 4.34.0
- pandas 2.0.0
- Pillow 9.0.0
- chardet 5.1.0

安裝所有依賴：
```bash
pip install -r requirements.txt
```

---

## **數據格式**

### **1. 訓練數據 (`train.txt`)**
訓練數據是一個 `.txt` 文件，包含兩列：`guid`（樣本唯一標識符）和 `label`（標籤）。文件格式如下：
```txt
guid    label
1       positive
2       neutral
3       negative
```

### **2. 驗證數據 (`val.txt`)**
驗證數據格式與訓練數據相同。

### **3. 測試數據 (`test_without_label_cleaned.txt`)**
測試數據僅包含 `guid` 列，格式如下：
```txt
guid
1
2
3
```

### **4. 文本和圖像數據**
- 每個樣本包含一個文本文件（如 `1.txt`）和一個圖像文件（如 `1.jpg`）。
- 文本文件存放樣本對應的文字內容。
- 圖像文件存放樣本對應的圖片內容。

---

## **模型架構**

1. **文本模態**：
   - 使用 BERT (`bert-base-uncased`) 提取文本特徵。
   - 特徵維度：`768`，經過全連接層降維到 `256`。

2. **圖像模態**：
   - 使用 ResNet18 提取圖像特徵。
   - 特徵維度：`512`，經過全連接層降維到 `256`。

3. **多模態融合**：
   - 將文本和圖像特徵拼接後經過全連接層輸出。
   - 輸出維度：`3`（`positive`, `neutral`, `negative`）。

---

## **執行流程**

### **1. 訓練模型**
運行以下命令進行模型訓練：
```bash
python train.py
```
訓練完成後，模型會保存為 `best_multimodal_model.pth`。

---

### **2. 測試模型**
運行以下命令進行測試：
```bash
python train.py
```
測試結果將保存到 `test_results.txt`，格式如下：
```txt
guid    label
1       positive
2       neutral
3       negative
```

---

## **參考實現**

1. [PyTorch](https://pytorch.org/)：作為深度學習框架。
2. [Transformers](https://huggingface.co/transformers/)：用於加載和處理 BERT 模型。
3. [torchvision](https://pytorch.org/vision/stable/)：加載 ResNet18 模型和圖像處理。
4. [Pandas](https://pandas.pydata.org/)：用於數據處理。
5. [Pillow](https://python-pillow.org/)：用於圖像讀取。

---

## **注意事項**

1. 確保所有數據文件（如 `train.txt`, `val.txt`, `test_without_label_cleaned.txt`，以及每個樣本的文本和圖像文件）存放在 `data/` 目錄中。
2. 在運行代碼前，請確認依賴環境已正確安裝。
3. 如果測試過程中出現錯誤，請檢查測試數據是否與格式一致。

---

## **反饋**

如有任何問題，請提交 [Issue](https://github.com/你的GitHub用戶名/你的項目名/issues)。

---

## **示例輸出**

訓練過程中的輸出示例：
```txt
使用設備：cuda
Epoch 1/10, Loss: 0.5678, Train Accuracy: 0.8234
Validation Accuracy: 0.8123
保存最佳模型至 best_multimodal_model.pth
...
```

測試結果將保存在 `test_results.txt`：

```txt
guid    label
1       positive
2       neutral
3       negative
```

---

你可以直接將此 README 文件保存為 `README.md`，並上傳到 GitHub。需要進一步調整或補充，隨時告訴我！