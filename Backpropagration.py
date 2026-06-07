import numpy as np
import matplotlib.pyplot as plt

class Backpropagation:
    def __init__(self, alpha=0.3, epoch=100, target_error=0.01):
        # Inisialisasi parameter jaringan
        self.alpha = alpha
        self.target_error = target_error
        self.epoch = epoch
        
        # Inisialisasi bobot awal secara acak (Sesuai contoh modul)
        # 2 input ke 2 hidden neuron, dan 2 hidden neuron ke 1 output
        self.w_hidden = np.array([[0.1, 0.4], [-0.2, 0.2]])
        self.b_hidden = np.array([[0.1, 0.2]])
        self.w_output = np.array([[0.2], [-0.5]])
        self.b_output = np.array([[-0.1]])

    def fungsi_aktivasi(self, x):
        # Menggunakan fungsi Bipolar Sigmoid sesuai modul
        return (2 / (1 + np.exp(-x))) - 1

    def turunan_aktivasi(self, f_x):
        # Turunan fungsi Bipolar Sigmoid
        return 0.5 * (1 + f_x) * (1 - f_x)

    def plot_decision_boundary(self, X, t, epoch):
        # Membuat scatter plot data input dengan warna berdasarkan target
        plt.scatter(X[:, 0], X[:, 1], c=t.ravel(), marker='o', edgecolors='k', cmap=plt.cm.RdYlBu, s=100)
        
        # Menentukan limit tampilan bidang grafik
        x_min, x_max = X[:, 0].min()-1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min()-1, X[:, 1].max() + 1
        
        # Membuat grid untuk menampilkan region keputusan
        h = 0.02  # step size di mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Prediksi output pada setiap titik di grid
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x_input = np.array([xx[i, j], yy[i, j]])
                z_net = np.dot(x_input, self.w_hidden) + self.b_hidden
                z = self.fungsi_aktivasi(z_net)
                y_net = np.dot(z, self.w_output) + self.b_output
                y = self.fungsi_aktivasi(y_net)
                Z[i, j] = y[0]
        
        # Plot contour
        plt.contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdYlBu, alpha=0.3)
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Decision Boundary Backpropagation XOR (Epoch {epoch+1})")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

    def fit(self, X, t):
        # Membuka file untuk mencatat log iterasi sesuai modul
        with open("Log_Pembelajaran_JST.txt", "w") as f:
            f.write("=== LOG PROSES PEMBELAJARAN JST BACKPROPAGATION ===\n\n")
            
            errors_per_epoch = []
            
            for epoch in range(self.epoch):
                f.write(f"--- EPOCH {epoch + 1} ---\n")
                total_error = 0
                
                for i in range(len(X)):
                    xi = X[i]
                    ti = t[i]
                    
                    # 1. Feedforward (Hidden Layer)
                    z_net = np.dot(xi, self.w_hidden) + self.b_hidden
                    z = self.fungsi_aktivasi(z_net)
                    
                    # Feedforward (Output Layer)
                    y_net = np.dot(z, self.w_output) + self.b_output
                    y = self.fungsi_aktivasi(y_net)
                    
                    # Hitung Error kuadratik
                    error = 0.5 * np.sum((ti - y) ** 2)
                    total_error += error
                    
                    # 2. Backpropagation (Output Layer)
                    error_output = ti - y
                    d_y = error_output * self.turunan_aktivasi(y)
                    
                    # Backpropagation (Hidden Layer)
                    error_hidden = np.dot(d_y, self.w_output.T)
                    d_h = error_hidden * self.turunan_aktivasi(z)
                    
                    # 3. Perbaikan Bobot dan Bias
                    self.w_output += np.dot(z.T, d_y) * self.alpha
                    self.b_output += np.sum(d_y, axis=0, keepdims=True) * self.alpha
                    self.w_hidden += np.dot(xi.reshape(2, 1), d_h) * self.alpha
                    self.b_hidden += np.sum(d_h, axis=0, keepdims=True) * self.alpha
                    
                    # --- BAGIAN KOREKSI: Catat detail perubahan bobot sesuai modul halaman 13 ---
                    f.write(f"Data ke-{i+1} -> Target: {ti.item()}, Output: {y.item():.4f}, Error: {error:.4f}\n")
                    f.write(f"Bobot output layer baru (w_output):\n{self.w_output}\n")
                    f.write(f"Bias output layer baru (b_output):\n{self.b_output}\n")
                    f.write(f"Bobot hidden layer baru (w_hidden):\n{self.w_hidden}\n")
                    f.write(f"Bias hidden layer baru (b_hidden):\n{self.b_hidden}\n")
                    f.write("-----------------------------------\n")
                
                errors_per_epoch.append(total_error)
                f.write(f"Total Error pada Epoch {epoch + 1}: {total_error:.4f}\n\n")
                
                # Tampilkan grafik simulasi setiap epoch 
                # (jika ingin skip ke hasil, disabled baris dibawah ini dengan tambah pagar di awal kalimat)
                self.plot_decision_boundary(X, t, epoch)
                
                # Cek jika error sudah di bawah target
                if total_error <= self.target_error:
                    f.write(f"Pembelajaran dihentikan. Target error tercapai pada epoch {epoch + 1}.\n")
                    print(f"[Sukses] Target error tercapai pada epoch {epoch + 1}")
                    break
                    
        print("Proses training selesai. Log dicatat di 'Log_Pembelajaran_JST.txt'")
