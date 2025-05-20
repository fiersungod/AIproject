# 專題名稱：AI分析惡意封包

本專題旨在運用人工智慧分析網路封包來辨別是否有惡意流量，目標是能夠檢測到惡意網路流量，以供後續的應對。

## 使用的技術：GAT + VAE

將網路封包轉換為Graph資料結構，使用GAT(Graph Attention Network)進行訓練就可以標註出封包之間的關聯(注意力係數)，藉此能夠使VAE獲得更好的訓練效果。

VAE(Variational AutoEncoder)利用將資料壓縮還原的方法，比較其輸入輸出。正常資料的有比較好的還原效果，但異常資料就會出現較大的失真，透過這種方法來達成偵測異常封包。

## 實驗環境

- python : 3.10.16
- numpy : 1.26.4
- pytorch : 2.2.2
- torch_cluster : 1.6.3
- torch-geometric : 2.6.1
- seaborn : 0.13.2

## 使用的訓練/測試資料

- 自行錄製的資料
- DrDoS2019 (UDP)
- UNSW-NB15