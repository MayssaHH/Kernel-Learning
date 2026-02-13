import torch
from torch.distributions import MultivariateNormal
def generate_mixture_of_gaussians(number_of_classes: int = 2 , samples_per_class: int = 100, dimension: int =2 , list_of_means: list = None, list_of_covariances: list = None):
    """
    Generate synthetic data from a mixture of Gaussians.
    
    Args:
        number_of_classes: Number of Gaussian components (classes).
        samples_per_class: Number of samples to generate for each class.
        dimension: Dimensionality of the input space.
        list_of_means: Optional list of means for each Gaussian component. If None, random means will be generated.
        list_of_covariances: Optional list of covariance matrices for each Gaussian component. If None, identity matrices will be used.

    Returns:
        X: Tensor of shape (number_of_classes * samples_per_class, dimension) containing the generated samples.
        y: Tensor of shape (number_of_classes * samples_per_class,) containing the class labels for each sample.
    """
    ## generate samples_per_class samples for each class using list_of_means[i] and list_of_covariances[i] for i in range(number_of_classes)
    if list_of_means is None:
        list_of_means = [torch.randn(dimension) * 5 for _ in range(number_of_classes)]
    if list_of_covariances is None:
        list_of_covariances = [torch.eye(dimension) for _ in range(number_of_classes)]
    
    X = []
    Y = []
    for i in range(number_of_classes):
        dist = MultivariateNormal(list_of_means[i],list_of_covariances[i])
        samples = dist.sample((samples_per_class,))
        X.append(samples)
        ## hence we have appended samples_per_class samples for class i to X 
        ## now to put the corresponding labels in Y , we can append i for samples_per_class times to Y
        Y.append(torch.full((samples_per_class,),i,dtype=torch.long))
    #print("before cat X:", X,"\n \n \n")
    X = torch.cat(X, dim=0)
    #print("after cat X:", X)
    #print("\n \n \n")
    print("before cat Y:", Y,"\n \n \n")
    #print("\n \n \n")
    Y = torch.cat(Y, dim=0)
   # print("after cat Y:", Y,"\n \n \n")
    return X, Y

if __name__ == "__main__":
    ## generate and visualize mixture of gaussians data , 2 classes , means -5 and 5 , identity covariance
    import matplotlib.pyplot as plt
    X, Y = generate_mixture_of_gaussians(number_of_classes=2, samples_per_class=10, dimension=2, list_of_means=[torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])], list_of_covariances=[torch.eye(2)*0.2, torch.eye(2)*0.2])
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
    plt.title("Mixture of Gaussians")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
