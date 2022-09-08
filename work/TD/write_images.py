import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np

image_width = 12
image_height = 4

def main():
    lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    df = np.array([np.genfromtxt(f"./data/Q_lambda_{_lambda}.csv", delimiter=",") for _lambda in lambdas])
    num_lambdas, num_episodes, num_states = df.shape
    
    fig = plt.figure(figsize=(25, 5))
    for j in range(num_episodes):
        for i in range(num_lambdas):
            img = df[i, j].reshape(image_height, -1)
            fig.add_subplot(1, num_lambdas, i+1)
            plt.axis("off")
            plt.title(f"λ = {lambdas[i]}")
            plt.imshow(img)
        fig.savefig(f"./images/heat_{j}.png")
        fig.clear()

if __name__ == '__main__':
    main()