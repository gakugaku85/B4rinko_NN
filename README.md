# Neural Network
NNの実装をまとめました。Python3の環境が整ったLinuxサーバで動かす際は基本的に以下のコマンドですべてのPythonファイルの実行がなされます。

> bash exe.sh

（注）これらのファイルはLinuxサーバで動かすことを想定して作りましたが、Linuxサーバに接続できない場合は、requirement.txtを参照し、必要なモジュールを取り入れ、"exe.sh"を参照し、その処理内容の順にプログラムを実行すればできると思います。

### nn.py
irisデータセットの3つのクラスの内、"setosa"と"versicolor"の2クラスを用い、neural networkによってテストデータを推定します。重みとバイアスの更新は、解析的微分を用いたものと、数値的微分を用いたものの２通りで実行し、それぞれの結果は"result"ディレクトリに".csv"として、また、Accuracy（精度）及びLoss（損失）の各エポックごとの推移を示すグラフを".png"として出力します。さらに、最終エポックでの重みとバイアスの更新値を".txt"で同ディレクトリ内へ出力します。
追記、重みとバイアスの計算を数値的微分だけにしました。またtrainingとvalidationにデータを分けて表示するようにしました（関口）

### dnn.py
mnistデータセットをdeep neural networkによってテストデータを推定します。mnistのデータを可視化したものを"result"ディレクトリに"mnist.png"として出力します。結果は同"result"ディレクトリに".csv"として、AccuracyとLossの推移についても".png"として出力されます。

実行方法をargsで入力できるようにしました（関口）
resultファイルをタイムスタンプでファイル訳するようにして、argsをtxtで出力するようにしました

### sample
```
python3 dnn.py -b 100 -eta 0.1 -e 100 -train_num 50000 -vali_num 10000 -test_num 10000
```

## 参考URL
"NumPyでニューラルネットワークを実装してみる 基本編"  
 https://deepage.net/features/numpy-neuralnetwork-1.html  
"NumPyでニューラルネットワークを実装してみる 理論編"  
https://deepage.net/features/numpy-neuralnetwork-2.html  
"NumPyでニューラルネットワークを実装してみる 実装編"  
https://deepage.net/features/numpy-neuralnetwork-3.html  
"NumPyでニューラルネットワークを実装してみる 多層化と誤差逆伝播法編"  
https://deepage.net/features/numpy-neuralnetwork-4.html  
"NumPyでニューラルネットワークを実装してみる 文字認識編"  
https://deepage.net/features/numpy-neuralnetwork-5.html


追記
パワポやプログラムは「ゼロから作る　Deep Learning pythonで学ぶディープラーニングの理論と実装」という本で書かれているものが多いです（関口）
