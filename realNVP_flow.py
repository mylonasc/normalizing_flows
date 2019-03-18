import tensorflow as tf
import numpy as np

def make_st(io_size, midsize = 20, n_interm = 1, activation = tf.nn.leaky_relu):
    """
    a function to make the scale and shift neural networks for the realNVP.
    The scale and shift transformations are implemented as dense networks (in the
    original paper they were CNNs)

    This is an alternative to the tensorflow implementation of "default_templates" for nvp bijectors.
    """
    s = tf.keras.Sequential()
    s.add(tf.keras.layers.Dense(midsize,input_dim = io_size))
    s.add(tf.keras.layers.Activation(activation))
    for k in range(0,n_interm):
        s.add(tf.keras.layers.Dense(midsize,input_dim = midsize))
        s.add(tf.keras.layers.Activation(activation))

    s.add(tf.keras.layers.Dense(io_size,input_dim = midsize))
    s.add(tf.keras.layers.Activation(activation))

    t = tf.keras.Sequential()
    t.add(tf.keras.layers.Dense(midsize,input_dim = io_size))
    t.add(tf.keras.layers.Activation(activation))
    for k in range(0,n_interm):
        t.add(tf.keras.layers.Dense(midsize,input_dim = midsize))
        t.add(tf.keras.layers.Activation(activation))

    t.add(tf.keras.layers.Dense(io_size,input_dim = midsize))
    t.add(tf.keras.layers.Activation(activation))
    return s,t


class nvp_flow(object):
    def __init__(self, s,t,change_ids):
        self.s = s;
        self.t = t;
        self.change_ids = change_ids;

        self.input_shape = int(s.input.shape[1] + t.input.shape[1])
        
        self.ids_unchanged = [k for k in range(0,self.input_shape) if k not in change_ids]

        self.index_order = [];
        self.index_order.extend(self.ids_unchanged)
        self.index_order.extend(self.change_ids);

    def get_split(self,x, reverse = False):
        x_unchanged  = tf.stack([x[:,k] for k in self.ids_unchanged],axis = 1)
        x_changed= tf.stack([x[:,k] for k in self.change_ids],axis = 1)
        out = [x_unchanged,x_changed] if not reverse else [x_changed,x_unchanged]
        
        return out

    def forward(self,x):
        x1,x2 = self.get_split(x)
        y1  = x1 
        y2  = x2 * tf.exp(self.s(x1)) + self.t(x1)
        y = [y1,y2]
        yy = tf.concat([y[k] for k in self.index_order],1)
        return yy

    def inverse(self,y):
        y1,y2 = self.get_split(y)
        x1  = y1;
        x2  = (y2 - self.t(x1)) *  tf.exp(-self.s(x1))
        xx = [x1,x2];
        xx = tf.concat([xx[k] for k in self.index_order],1)
        #print('calling inverse %s'%self)
        return xx

    def log_det_jac(self,x):
        x_unchanged,x_changed = self.get_split(x)
        # assuming the first dimension is the batch dimension
        return tf.reduce_sum(self.s(x_unchanged),axis = 1) 

    def inv_log_det_jac(self,x):
        # jacobian of the inverse is inverse of jacobian
        x_unchanged,x_changed = self.get_split(x);
        return -tf.reduce_sum(self.s(x_unchanged),axis = 1)

    def det_jac(self,x):
        return tf.exp(self.log_det_jac(x))

    def inv_det_jac(self,x):
        return tf.exp(self.inv_log_det_jac(x))



class nvp_stack(object):
    def __init__(self,nlayers,flow_output, width_st = 10, n_interm = 1, st_activation = tf.nn.relu):
        """ 
        the flow output should be the variable that we don't know the distribution of.
        the "input" - typically a latent "z" of a simple distribution, as far as the 
        flows are concerned is not relevant.

        They flow stack is computed as a "chain" from flow_output to "z", which is
        expected to be easy to fit with a simple prior (a diagonal gaussian, a 
        mixture of gaussians and the like).

        @width_st: the width of the layers for scale and shift transformation neural networks
        @n_interm: the number of intermediate layers of the scale and shift transformation

        """

        self.s = [];
        self.t = [];
        self.nvp_layers = [];
        self.flow_output = flow_output;

        curr_var = flow_output;

        # notice that the layers are to be applied in reverse order!
        for k in range(0,nlayers):
            s_,t_ = make_st(io_size = 1, midsize = width_st, n_interm = n_interm, activation = st_activation)
            self.s.append(s_)
            self.t.append(t_)
            change_ids = [0] if k%2==0 else [1] # Alternating the 2 dimensions of each flow.
            new_flow = nvp_flow(s_,t_,change_ids)
            curr_var = new_flow.inverse(curr_var)
            self.nvp_layers.append(new_flow)
            

        # the "layer ordering" convention is increasing from z to x. (that's why apply the inverses above)
        # The lists are reversed

        self.s = self.s[::-1]
        self.t = self.t[::-1]
        self.nvp_layers = self.nvp_layers[::-1]
        self.base_var = curr_var
        #the last curr_var is the variable in the space of the base distribution.

    def inverse(self):
        """
        a pointer to the variable is saved during construction.
        """
        # the nvp layers are defined from z to x (no need to reverse order)
        
        return self.base_var

    def forward(self,z):
        """
        in general z is not known. 
        The inverse transformations are usually applied to a variable to get to "z"
        At the "z" space we can easilly compute likelihoods (we define a simple 
        distribution there). The purpose of the flow is to warp the space so that 
        the datapoints have a high likelihood under a imple prior"  z.

        This function is used when transforming from "z" to "x".
        """
        z_next = z

        # The reversed order is the order from "z" to "x":
        for nvp_layer in self.nvp_layers:
            z_next = nvp_layer.forward(z_next)

        return z_next

    def inv_log_det_jac(self,x = None):
        """
        computes the inverse of the whole transformation for variable "x"
        @x tf.Tensor or np.array of appropriate size. If x == None, then 
            the input tensor is used for computation.
        """
        if x is None:
            x = self.flow_output

        sld = np.float32(0.);
        x_in = x;
        for nvp_layer in reversed(self.nvp_layers):
            sld = sld + nvp_layer.inv_log_det_jac(x_in)
            x_in = nvp_layer.inverse(x_in)

        #code.interact(local = dict(locals(), **globals()))

        return sld

    def inv_dets(self,x):
        # start with "unit volume" and apply the inverse determinants 
        # consequtively to get transformed volume:
        prd_det = np.float32(1.) ;# tf.cast( 1.,dtype = tf.float32) ; #tf.ones(x.shape[0], dtype = tf.float32); 

        if x is None:
            x = self.flow_output

        x_in = x;
        for nvp_layer in reversed(self.nvp_layers):
            prd_det = prd_det * nvp_layer.det_jac(x_in)

            x_in=nvp_layer[1](x_in)
            # code.interact(local = dict(locals(), **globals()))

        z = x_in;
        return prd_det 

    def prod_dets(self,x):
        """
        given "z", compute the produce of the determinants of the 
        series of the transformations in order to compute likelihood (not log)

        For debugging.
        """
        error('not implemented')




    def prob(self, x = None, base_dist = None):

        if x is None:
            x = self.flow_output

        prd_det ,z = self.inv_dets(x)
        return base_dist.prob(z) * prd_det

    def log_prob(self,x = None, base_dist = None):
        """
        Computes the log probabilities, also taking into account a base distribution.
        """

        if x is None:
            x = self.flow_output
            
        sum_logdets = self.inv_log_det_jac(x)
        return sum_logdets + base_dist.log_prob(self.base_var)
