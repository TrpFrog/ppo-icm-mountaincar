# drl-spring-final

このリポジトリは「DRLスプリングセミナー2023」の最終課題のリポジトリです。

## タスク

このリポジトリでは「Mountain-Car-v0」の環境を使用し、
PPO の Actor 側学習率、Critic 側学習率を変えたときの性能比較を行います。

## 動作環境

  - Python 3.10.0
  - macOS 13.2.1
  - Apple M1 Pro (16GB)
    - 学習は cpu で実施 

使用ライブラリについては requirements.txt の通りです。

## 動作方法

### 学習

```bash
python -m src --train
```

### テスト

```bash
python -m src --test
```

