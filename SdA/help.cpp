class dA {

public:
  int N;
  int n_visible;
  int n_hidden;
  double **W;
  double *hbias;
  double *vbias;
  dA(int, int, int , double**, double*, double*);
  ~dA();
  void get_corrupted_input(int*, int*, double);
  void get_hidden_values(int*, double*);
  void get_reconstructed_input(double*, double*);
  void train(int*, double, double);
  void reconstruct(int*, double*);
};


// dA
dA::dA(int size, int n_v, int n_h, double **w, double *hb, double *vb) {
  N = size;
  n_visible = n_v;
  n_hidden = n_h;

  if(w == NULL) {
    W = new double*[n_hidden];
    for(int i=0; i<n_hidden; i++) W[i] = new double[n_visible];
    double a = 1.0 / n_visible;

    for(int i=0; i<n_hidden; i++) {
      for(int j=0; j<n_visible; j++) {
        W[i][j] = uniform(-a, a);
      }
    }
  } else {
    W = w;
  }

  if(hb == NULL) {
    hbias = new double[n_hidden];
    for(int i=0; i<n_hidden; i++) hbias[i] = 0;
  } else {
    hbias = hb;
  }

  if(vb == NULL) {
    vbias = new double[n_visible];
    for(int i=0; i<n_visible; i++) vbias[i] = 0;
  } else {
    vbias = vb;
  }
}

dA::~dA() {
  // for(int i=0; i<n_hidden; i++) delete[] W[i];
  // delete[] W;
  // delete[] hbias;
  delete[] vbias;
}

void dA::get_corrupted_input(int *x, int *tilde_x, double p) {
  for(int i=0; i<n_visible; i++) {
    if(x[i] == 0) {
      tilde_x[i] = 0;
    } else {
      tilde_x[i] = binomial(1, p);
    }
  }
}

// Encode
void dA::get_hidden_values(int *x, double *y) {
  for(int i=0; i<n_hidden; i++) {
    y[i] = 0;
    for(int j=0; j<n_visible; j++) {
      y[i] += W[i][j] * x[j];
    }
    y[i] += hbias[i];
    y[i] = sigmoid(y[i]);
  }
}

// Decode
void dA::get_reconstructed_input(double *y, double *z) {
  for(int i=0; i<n_visible; i++) {
    z[i] = 0;
    for(int j=0; j<n_hidden; j++) {
      z[i] += W[j][i] * y[j];
    }
    z[i] += vbias[i];
    z[i] = sigmoid(z[i]);
  }
}

void dA::train(int *x, double lr, double corruption_level) {
  int *tilde_x = new int[n_visible];
  double *y = new double[n_hidden];
  double *z = new double[n_visible];

  double *L_vbias = new double[n_visible];
  double *L_hbias = new double[n_hidden];

  double p = 1 - corruption_level;

  get_corrupted_input(x, tilde_x, p);
  get_hidden_values(tilde_x, y);
  get_reconstructed_input(y, z);
  
  // vbias
  for(int i=0; i<n_visible; i++) {
    L_vbias[i] = x[i] - z[i];
    vbias[i] += lr * L_vbias[i] / N;
  }

  // hbias
  for(int i=0; i<n_hidden; i++) {
    L_hbias[i] = 0;
    for(int j=0; j<n_visible; j++) {
      L_hbias[i] += W[i][j] * L_vbias[j];
    }
    L_hbias[i] *= y[i] * (1 - y[i]);

    hbias[i] += lr * L_hbias[i] / N;
  }
  
  // W
  for(int i=0; i<n_hidden; i++) {
    for(int j=0; j<n_visible; j++) {
      W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / N;
    }
  }

  delete[] L_hbias;
  delete[] L_vbias;
  delete[] z;
  delete[] y;
  delete[] tilde_x;
}

void dA::reconstruct(int *x, double *z) {
  double *y = new double[n_hidden];

  get_hidden_values(x, y);
  get_reconstructed_input(y, z);

  delete[] y;
}
