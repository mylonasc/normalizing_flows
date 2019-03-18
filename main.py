import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy  as np
import matplotlib.pyplot as pplot
import code

import sklearn.datasets
from utils import *

from realNVP_flow import *

points = sklearn.datasets.make_moons(1000, noise = 0.04);

# flow model for two moons datasets:
sess = tf.InteractiveSession()

dats = points[0];# np.random.randn(100,2)*0.5  + np.array([-1,1])


ngrid = 50;
[x,y] = np.meshgrid(np.linspace(-2,2.5,ngrid),np.linspace(-2,2,ngrid))

if False:
    d1=tfd.MultivariateNormalDiag(loc = [0,0],  scale_diag = [1.,0.1])
    d2=tfd.MultivariateNormalDiag(loc = [0,0.5],scale_diag = [1.,0.1])
    d3=tfd.MultivariateNormalDiag(loc = [0, 1], scale_diag = [1.,0.1])
    d4=tfd.MultivariateNormalDiag(loc = [0, 0], scale_diag = [0.1,1.])
    d5=tfd.MultivariateNormalDiag(loc = [0.5,0],scale_diag = [0.1,1.])
    d6=tfd.MultivariateNormalDiag(loc = [1., 0],scale_diag = [0.1,1.])

    d = tfd.Mixture(cat = tfd.Categorical(probs=[1./6. for k in range(0,6)]),components= [d1,d2,d3,d4,d5,d6])

d1 = tfd.MultivariateNormalDiag(loc = [-1,0],  scale_diag = [0.5,0.5])
d2 = tfd.MultivariateNormalDiag(loc = [1,0],  scale_diag = [0.5,0.5])
d = tfd.Mixture(cat = tfd.Categorical(probs=[1./2. for k in range(0,2)]),components= [d1,d2])

#d = tfd.MultivariateNormalDiag(loc = [0.,0.],  scale_diag = [1,1])

vv_grid = np.hstack([x.reshape([-1,1]),y.reshape([-1,1])]);
z_prob = d.prob(vv_grid).eval();
x_in = tf.placeholder(shape = [None,2],dtype = tf.float32)

nflows = 4
width_st = 100
act_dict = {'relu' : tf.nn.relu, 'tanh' : tf.nn.tanh,'sigmoid' : tf.nn.sigmoid,'leaky_relu' : tf.nn.leaky_relu}
activation_st = 'leaky_relu'

nvp = nvp_stack(nflows, x_in, width_st = width_st, n_interm = 1, st_activation = act_dict[activation_st])

log_det_jacs = nvp.inv_log_det_jac(x_in)

# Take output datapoints, apply inverse transforms, and compute likelihood in base distr.
# the nvp.inverse is a handle to the inverse transform of "x" (i.e. after all the inverse transformations have been applied).
log_likelihood_base=  d.log_prob(nvp.inverse())

log_likelihood = log_likelihood_base + log_det_jacs 


nvplp = nvp.log_prob(base_dist = d)
opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize( - tf.reduce_mean(nvplp,axis = 0))
losses = [];

sess.run(tf.global_variables_initializer())

for k in range(1,2000):
    
    [log_prob_nvp_, opt_, ]  = sess.run([nvplp, opt], feed_dict  = {x_in : dats})

    losses.append(log_prob_nvp_);
    print('loss: %f'%np.sum(log_prob_nvp_))

    if k%100 == 0 or k == 0:

        # That's for inspection:
        [lprop_grid] = sess.run([nvplp], feed_dict  = {x_in : vv_grid})
        pplot.subplot(2,1,1); pplot.cla()
        plotgrid(vv_grid,z_prob)
        pplot.xlim([-1.5,2.5]) ; 
        pplot.ylim([-1,1]); 
        vals = nvp.inverse().eval(feed_dict = {x_in : dats}); pplot.plot(vals[:,0], vals[:,1],'o')
        pplot.title('base distribution & \n points in latent space')


        pplot.subplot(2,1,2);pplot.cla()
        ss = d.sample(100); mypoints = nvp.forward(ss).eval();
        plotgrid(vv_grid,z = np.exp(lprop_grid)); 
        pplot.xlim([-1.5,2.5]) ; 
        pplot.ylim([-1,1]); 
        pplot.show(block  = False)
        
        pplot.plot(dats[:,0],dats[:,1],'.', alpha = 0.2); 
        pplot.xlim([-1.5,2.5]) ; 
        pplot.ylim([-1,1]); 
        pplot.plot(mypoints[:,0],mypoints[:,1],'.'); 
        pplot.title('physical space (x) Nflows: %i W_st: %i Act: %s'%(nflows,width_st, activation_st ))
        
        pplot.show(block = 0 )
        pplot.pause(0.1);

        fig = pplot.gcf(); fig.set_size_inches(7.5, 7.5, forward=True)
        pplot.savefig('step_%03i.png'%k)
