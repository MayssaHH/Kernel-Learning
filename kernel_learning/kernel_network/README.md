# TWo approaches I took to do what we discussed a kernel:
## Reminder what we discussed
Have an arthitecture that uses linear kernels as a starting point and then combine them using KAN where we will have our "x" as our p kernels.

## (1) Description of the first approach
$$ K(u,v) = \sum_{i=1}^p \alpha_i^2 \Phi_i\left(\sum_{j=1}^p \phi_{i,j}\left(K(u^{(j)}, v^{(j)})\right)\right) \quad \quad \forall u, v \in \mathcal{X} \subset \mathbb{R}^p (1)$$
Where the $u^{(j)}$ and $v^{(j)}$ are the j-th components of the input vectors u and v, respectively. 

So the n x n kernel matrix K is computed as follows:
$$ K_{ab} = K(u_a, u_b) = \sum_{i=1}^p \alpha_i^2 \Phi_i\left(\sum_{j=1}^p \phi_{i,j}\left(K(u_a^{(j)}, u_b^{(j)})\right)\right) \quad \quad \forall a, b = 1, \ldots, n$$
Matrix: 
$$ K = \begin{bmatrix} K(u_1, u_1) & K(u_1, u_2) & \ldots & K(u_1, u_n) \\ K(u_2, u_1) & K(u_2, u_2) & \ldots & K(u_2, u_n) \\ \vdots & \vdots & \ddots & \vdots \\ K(u_n, u_1) & K(u_n, u_2) & \ldots & K(u_n, u_n) \end{bmatrix}$$

In other words, I am taking the KAN archtecture, but my inputs are not the raw input ($u, v \in \mathcal{X}$), but rather the kernel values computed on each dimension separately (i.e. the $K(u^{(j)}, v^{(j)})$ terms). So the input to the KAN architecture is a p-dimensional vector of kernel values, and the output is a single scalar which is the final kernel value for that pair of inputs. And I am modifying the KAN archtecture to have a learnable $\alpha_i^2$ term in front of each $\Phi_i$ term. 

Notice that the $\alpha_i^2$ term is sqaured and possibly normalized, which ensures that it is non-negative and thus preserves the positive definiteness of the kernel. BUT the $\alpha_i^2$ are NOT per feature weights but rather a combination of per feature kernels applied to them a non-linearity, and then summing them and then doing a non-linearity and after that applying the $\alpha_i^2$ weights. I will be looking soon at where to put the $\alpha_i^2$ weights to make it more "interpretable" in the sense of being able to say "this feature is more important than that one" based on the value of $\alpha_i^2$. 

Note: the $\alpha_i$ need not be squared, in my architecure I made so default is squared but you have other transformation you can apply here, check the [`KAN_kernel`](./KAN_Kernel.py) implementation. 

Note: also for now the $K(u^{(j)}, v^{(j)})$ are computed using a fixed kernel (Linear for now)

Note: curretnly I naively compute an n x n kernel matrix, a better approach to compute the upper triangular part and then mirror it to get the lower triangular part, to save some complexity of the code. I will implement this in the future.

I added a check to see if the results $n x n$ kernel matrix is symmetric and positive definite.

### Issues with the latter approach:
1. I created the architecture and it ran, but I noticed that is was always giving me the latter warning that it is NOT PSD. So I looked at the formula (1) and I realized that (Taken from the Intro to SVM book) it does not follow these closure properties of kernels, which are the following:
#### Proposition 3.12

Let $K_1$ and $K_2$ be kernels over $X \times X$, $X \subseteq \mathbb{R}^n$, $a \in \mathbb{R}^+$, $f(\cdot)$ a real-valued function on $X$,

$$\phi : X \longrightarrow \mathbb{R}^m$$

with $K_3$ a kernel over $\mathbb{R}^m \times \mathbb{R}^m$, and $\mathbf{B}$ a symmetric positive semi-definite $n \times n$ matrix. Then the following functions are kernels:

1. $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) + K_2(\mathbf{x}, \mathbf{z})$

2. $K(\mathbf{x}, \mathbf{z}) = a K_1(\mathbf{x}, \mathbf{z})$

3. $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) K_2(\mathbf{x}, \mathbf{z})$

4. $K(\mathbf{x}, \mathbf{z}) = f(\mathbf{x}) f(\mathbf{z})$

5. $K(\mathbf{x}, \mathbf{z}) = K_3(\phi(\mathbf{x}), \phi(\mathbf{z}))$

6. $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}' \mathbf{B} \mathbf{z}$

#### Reasons: 
1. f(kernel) is not always a kernel, it depends on the function f, I have read online that this is true if f can be experesses as a power series with non-negative coefficients, (which in the direct setup there is no guarantee for that)

#### Solution -> see second approach below.


## (2) Description of the second approach
I will be using 4th propery of kernels, which is the following:
$$K(\mathbf{x}, \mathbf{z}) = f(\mathbf{x}) f(\mathbf{z})$$
Where f is a real-valued function on X. 

So I will write: 
$$ \forall u, v \in \mathcal{X} \subset \mathbb{R}^p$$ and $$K: \mathcal{X} \times \mathcal{X} \longrightarrow \mathbb{R}$$  I will write the kernel as follows:

$ K(u, v) = \psi(u) \psi(v) $

Where $\psi(.)$ is our KAN network $$\psi(u): \mathcal{X} \to \mathbb{R}, $$ written as follows:

$$ \psi(u) = \sum_{i=1}^p  \Phi_i\left(\sum_{j=1}^p \phi_{i,j}\left(u^{(j)}\right)\right) \quad \quad \forall u \in \mathcal{X} \subset \mathbb{R}^p$$
Where the $u^{(j)}$ IS the j-th components of the input vector u.

### Learning Procedure: 
2 parallel runs to get $\psi(u)$ and $\psi(v)$, then compute the kernel value as $K(u, v) = \psi(u) \psi(v)$, then compute the loss(e.g. aligment) based on the kernel matrix computed on the training data(or a batch), and backpropagate to update the parameters of $\psi(.)$.

### Feature Selection (or interpretability)
I was thinking to do the following: 
$$\psi(u) = \sum_{i=1}^p  \Phi_i\left(\sum_{j=1}^p \phi_{i,j}\left( \alpha^{(j)}u^{(j)}\right)\right),$$

where the $\alpha^{(j)}$ are feature weights(learnable params) that are applied to the input features before applying the $\phi_{i,j}$ kernels. This way, the $\alpha^{(j)}$ can be interpreted as feature importance weights, and we can analyze their values to understand which features are more important in the learned kernel. With the constrained that: 
$$\sum_{j=1}^p \alpha^{(j)} = 1, \quad \alpha^{(j)} \geq 0 \quad \forall j = 1, \ldots, p$$

Or we can put the: alphas after the first layer so one alpha common between all the phi that are common for the feature j, so we can write:
$$\psi(u) = \sum_{i=1}^p  \Phi_i\left(\sum_{j=1}^p \alpha^{(j)}\phi_{i,j}\left(u^{(j)}\right)\right)$$

<!-- # TODO: 
1. Check args and kwargs in inference mode for SVC because it needs to compute the kernel matrix between X_support and X_test, so I need to change my architecture to be able to compute the kernel matrix between two different sets of inputs, not just between the training inputs. This is important for inference mode. SO I will be changing the methods to take into consideration (X_1, X_2) instead of just (X) for computing the kernel matrix. Before I was just computing the kernel matrix on the training data, but now I need to be able to compute it between any two sets of inputs (e.g. support and test).
2. Test the architecture on the data I have to see 
3. Fix the rest of the kernels to take two x's not just one x.
4. Fix the code for optimier to remove the lambda lasso penalty for now  -->
