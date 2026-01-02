# PAMED: Personalized Affective Modeling based on EEG Dynamics for Multimodal Music Generation

### EEG-MIDI-VA Dataset
**Dataset Structure**
```
dataset/
├──Music_Data
│   ├── music.xlsx
│   ├── 1.mid (or .midi)
│   ├── 2.mid (or .midi)
│   └── ...
└── User_Data
    ├── user.xlsx
    ├── sub_01
    │   ├── EEG
    │   │    ├── sub_01_IADS.cnt
    │   │    ├── sub_01_music_1.cnt
    │   │    └── sub_01_music_2.cnt
    │   ├── MIDI
    │   │    ├── exp_1_music_1_mid_xxx.mid (or .midi)
    │   │    ├── exp_1_music_2_mid_xxx.mid (or .midi)
    │   │    └── ...
    ├── sub_02
    │   ├── EEG
    │   └── MIDI
    ├── ...
```


feature(.h5 file):
```
文件中的数据集和组：
数据集: alpha_power, 形状: (68,), 数据类型: float64
数据集: beta_power, 形状: (68,), 数据类型: float64
数据集: delta_power, 形状: (68,), 数据类型: float64
数据集: gamma_power, 形状: (68,), 数据类型: float64
数据集: mean, 形状: (68,), 数据类型: float64
数据集: std, 形状: (68,), 数据类型: float64
数据集: theta_power, 形状: (68,), 数据类型: float64
数据集: wavelet_entropy, 形状: (68,), 数据类型: float64
```