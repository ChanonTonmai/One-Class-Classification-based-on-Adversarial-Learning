import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
np.random.seed(1340)

'''
Need to define you own x_train, x_test, and y_test
'''

#%%
# Dimensionality Reduction
from sklearn.decomposition import KernelPCA 
transformer = KernelPCA(n_components=100,kernel='rbf')
x_t_train = transformer.fit_transform(x_train)
x_t_test = transformer.fit_transform(x_test)

#%%
from keras import initializers
def build_generator(size):
    input_data = Input(shape=(size,))
    x = Dense(4096, activation='relu',kernel_initializer=initializers.he_uniform(seed=None))(input_data)
    x = BatchNormalization()(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(size, activation='relu')(x)
    x = BatchNormalization()(x)
    return Model(input_img, x, name='R')

def build_discriminator(size):

    input_data = Input(shape=(size,))
    x2 = Dense(4096, activation='relu')(input_data)
    x2 = BatchNormalization()(x2)
    x2 = Dense(2048, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(512, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(128, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(1, activation='relu')(x2)
    return Model(input_img2, x2, name='D')


optimizer = RMSprop(lr=0.0002, clipvalue=1.0, decay=1e-9)

discriminator = build_discriminator()

# Model to train D to discrimate real images.
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

# Construct generator/R network.
generator = build_generator()
image_dims = [100]
img = Input(shape=image_dims)

reconstructed_img = generator(img)
discriminator.trainable = False
validity = discriminator(reconstructed_img)

# Model to train Generator/R to minimize reconstruction loss and trick D to see
# generated images as real ones.
adversarial_model = Model(img, [reconstructed_img, validity])
adversarial_model.compile(loss=['binary_crossentropy','binary_crossentropy'],
    optimizer=optimizer)
discriminator.summary()
adversarial_model.summary()


np.random.seed(seed_value)
noise = np.random.normal(0, 1, x_t_train.shape)
x_noise = x_t_train + noise 

epochs = 20
batch_size = 32
batch_idxs = len(x_t_train) // batch_size

ones = np.ones((batch_size, 1))
zeros = np.zeros((batch_size, 1))

for epoch in range(epochs):
    print('Epoch ({}/{})-------------------------------------------------'.format(epoch,epochs))
    for idx in range(0, batch_idxs):
        batch = x_t_train[idx*batch_size:(idx+1)*batch_size]
        batch_noise = x_noise[idx*batch_size:(idx+1)*batch_size]
        batch_clean = x_t_train[idx*batch_size:(idx+1)*batch_size]
        
        batch = np.array(batch).astype(np.float32)
        batch_noise = np.array(batch_noise).astype(np.float32)
        batch_clean = np.array(batch_clean).astype(np.float32)
        
        batch_fake = generator.predict(batch_noise)
        # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
        np.random.seed(seed_value)
        d_loss_real = discriminator.train_on_batch(batch, ones)
        d_loss_fake = discriminator.train_on_batch(batch_fake, zeros)

        np.random.seed(seed_value)
        # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
        adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
        g_loss = adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
        msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{4:>0.3f}'.format(epoch, idx, batch_idxs, d_loss_real+d_loss_fake, g_loss[0], g_loss[1])
    print(msg)
        
#%% 
fraud_pred = (adversarial_model.predict(x_t_test))
from sklearn.cluster import KMeans
from sklearn import mixture

kmeans = KMeans(n_clusters=2, random_state=0).fit(fraud_pred[1])
y_pred = kmeans.labels_
#
#%%
#y_pred = np.where(y_pred<0.5,0,1)
TP = FN = FP = TN = 0
for j in range(len(y_test)):
    if y_test[j] == 0 and y_pred[j] == 1:
        FP = FP+1
    elif y_test[j] == 0 and y_pred[j] == 0:
        TN=TN+1
    elif y_test[j] == 1 and y_pred[j] == 1:
        TP=TP+1
    else:
        FN=FN+1

print (TP,  FN,  FP,  TN)
accuracy = (TP+TN)/(TP+FN+FP+TN)
FPR = 100*FP/(FP+TN)    
FNR = 100*FN/(FN+TP)
print('False positive rate from KMean: ',FPR)
print('False negative rate from KMean: ',FNR) 
print('Acc: ',accuracy)
