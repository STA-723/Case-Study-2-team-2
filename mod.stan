functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau)
                    + sum(ldet_terms)
                    - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}

data {
  int<lower = 1> n;
  int<lower = 1> p;
  matrix[n, p] X;
  vector[n] y;
  int W_n;
  int<lower = 1> n_nbhd;
  int<lower = 1> n_bur;
  int<lower = 1> nbhd[n];
  int<lower = 1> bur[n];
  matrix<lower = 0, upper = 1>[n_nbhd, n_nbhd] W;
}
transformed data{
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[n_nbhd] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[n_nbhd] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(n_nbhd - 1)) {
      for (j in (i + 1):n_nbhd) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1:n_nbhd) D_sparse[i] = sum(W[i]);
  {
    vector[n_nbhd] invsqrtD;  
    for (i in 1:n_nbhd) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}
parameters {
  vector[p] beta;
  vector[n_nbhd] phi;
  real<lower = 0> tau;
  real<lower = 0> tau_nbhd;
  real<lower = 0> tau_bur;
  real<lower = 0, upper = 1> alpha;
  vector[n_bur] beta_bur;
}
model {
  vector[n] effect_bur;
  vector[n] effect_nbhd;
  phi ~ sparse_car(tau_nbhd, alpha, W_sparse, D_sparse, lambda, n_nbhd, W_n);
  beta ~ normal(0, 25);
  beta_bur ~ normal(0, 1 / pow(tau_bur,.5));
  //beta_bur ~ normal(0,25);
  tau_nbhd ~ gamma(.1, .1);
  tau_bur ~ gamma(.1, .1);
  tau ~ gamma(.1, .1);
  for (i in 1:n) {
    effect_bur[i] = beta_bur[bur[i]];
    effect_nbhd[i] = phi[nbhd[i]];
  }
  y ~ normal(X * beta + effect_nbhd + effect_bur, inv(square(tau)));
}









