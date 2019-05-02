from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU # max(0.01x, x)b
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG
import matplotlib.pyplot as plt
import sys
import numpy as np


class GAN():
    def __init__(self, latent_dim=100, optimizer=None, img_dir="images/gan"):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.img_dir=img_dir
        
        if not optimizer:
            # Adam: Adaptive Grad + Momentum
            # learning rate, beta_1
            optimizer = Adam(0.0002, 0.5)
        
        # Discriminatorを作成し，コンパイル
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # Generatorを作成
        self.generator = self.build_generator()
                
        # GeneratorとDiscriminatorからなるCombined Network
        # Combined Networkは，GeneratorがDiscriminatorを騙せるよう訓練する
        self.combined = self.build_combined(self.generator, self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim)) #100in 256out
        model.add(LeakyReLU(alpha=0.2)) # activation
        model.add(BatchNormalization(momentum=0.8)) # boost
        model.add(Dense(512)) # 256in 512out
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024)) # 512in 1024out
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh')) # 画像データベクトルの生成
        model.add(Reshape(self.img_shape)) # 生成されたベクトルを画像の形に変形
        model.summary()
        return model
    
    def build_discriminator(self):
        model = Sequential()        
        # DiscriminatorにBatchNormを入れると強くなりすぎるので入れないらしい
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid')) # sigmoidで確率を出力
        model.summary()
        return model
    
    def build_combined(self, generator, discriminator):
        # DiscriminatorはGeneratorの学習時，学習しない
        # Trueに戻していないので不安になるが，compileしないと更新されないので大丈夫
        discriminator.trainable = False
        return Sequential([generator, discriminator])
    
    def train(self, epochs, batch_size=128, save_interval=50):
        
        # mnistのデータ読み込み
        (X_train, _), (_, _) = mnist.load_data()
        
        # 値を-1 ~ 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        half_batch = int(batch_size / 2)
        
        for epoch in range(epochs+1):
            
            # -------------------------
            # Discriminatorの学習
            # -------------------------
            
            # バッチサイズの半数をGeneratorから生成
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            # バッチサイズの半数を教師データからピックアップ
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            
            # discriminatorを学習
            # 本物データと偽物データは別々に学習させる
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            
            # それぞれの損失関数を平均
            d_loss= 0.5 * np.add(d_loss_real, d_loss_fake)
            
            
            # -----------------------
            # Generatorの学習
            # -----------------------
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # 生成データの正解ラベルは本物(1)
            valid_y = np.ones(batch_size, dtype=int)
            
            # Generatorを学習
            g_loss = self.combined.train_on_batch(noise, valid_y)
            
            # 指定した間隔で生成画像を保存 & 進捗出力
            if epoch % save_interval == 0:
                # 進捗の表示
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" \
                   % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                
                self.save_imgs(epoch)
    
    # 指定epochでの画像を保存
    def save_imgs(self, epoch):
        r,c = 5,5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        # Rescale images 0 ~ 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        
        fig.savefig(f"{self.img_dir}/{epoch}.png")
        plt.close()
    
    def visualize(self, model):
        dot = model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')
        return SVG(dot)
    
    def visualize_discriminator(self):
        print("Visualize: Discriminator")
        return self.visualize(self.discriminator)
    
    def visualize_generator(self):
        print("Visualize: Generator")
        return self.visualize(self.generator)
    
    def visualize_combined(self):
        print("Visualize: Combined")
        return self.visualize(self.combined)