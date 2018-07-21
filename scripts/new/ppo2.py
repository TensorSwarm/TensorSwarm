# Copyright (c) 2017 OpenAI (http://openai.com)
# Copyright (C) 2018 deeplearningrobotics.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import time
import joblib
import tensorboard_logging
import numpy as np
import os.path as osp
import tensorflow as tf
from collections import deque
import logger
from numba import jit

class AbstractEnvRunner(object):
    def __init__(self, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        raise NotImplementedError

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, deterministic=False):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False, deterministic=deterministic)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True, deterministic=deterministic)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def reshape2d(arr):
            return np.reshape(arr, (len(arr)*len(arr[0]), len(arr[0][0])), 'F')

        def reshape1d(arr):
            return np.reshape(arr, (len(arr)*len(arr[0])), 'F')

        def reshape(ids, obs, returns, masks, actions, values, neglogpacs):

            ib = np.asarray([[o["id"] for o in iobs] for iobs in obs])
            lb = np.asarray([[o["laser"] for o in iobs] for iobs in obs])
            rb = np.asarray([[o["rel_goal"] for o in iobs] for iobs in obs])
            vb = np.asarray([[o["velocities"] for o in iobs] for iobs in obs])

            lb = np.reshape(lb, (len(lb)*len(lb[0]), 512, 3), 'F')

            ib = reshape2d(ib)
            rb = reshape2d(rb)
            vb = reshape2d(vb)

            ids = reshape1d(ids)

            actions = reshape2d(actions)

            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            advs = reshape1d(advs)
            returns = reshape1d(returns)
            neglogpacs = reshape1d(neglogpacs)
            values = reshape1d(values)

            return ib, lb, rb, vb, advs, returns, masks, actions, values, neglogpacs

        def train(lr, cliprange, ids, obs, returns, masks, actions, values, neglogpacs, states=None):


            ib, lb, rb, vb, advs, returns, masks, actions, values, neglogpacs = reshape(ids, obs, returns, masks, actions, values, neglogpacs)

            td_map = {train_model.laser:lb,train_model.rel_goal:rb, train_model.velocities:vb, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            save_path = self.saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)

        def restore(restore_path):
            self.saver.restore(sess, restore_path)
            print("Model restored.")


        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.saver = tf.train.Saver()
        self.save = save
        self.restore = restore
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, gamma, lam):
        super(Runner, self).__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.env.set_model(self.model)
    def run(self):
        mb_ids, mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        self.ids, self.obs = self.env.reset()
        self.dones = [False] * self.env.num_robots
        for _ in range(self.nsteps):
            #print self.obs
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_ids.append(self.ids)
            mb_obs.append(self.obs)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.ids, self.obs, rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        #mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_ids = np.asarray(mb_ids)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 #- self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 #- mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return mb_ids, mb_obs, sf01(mb_returns), sf01(mb_dones), sf01(mb_actions), sf01(mb_values), sf01(mb_neglogpacs), \
                mb_states, epinfos
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    return arr
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, restore_path=None, deterministic=False):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    time_str = time.strftime("%m_%m_%H_%M_%S")
    board_logger = tensorboard_logging.Logger(os.path.join(logger.get_dir(), "tf_board", time_str))

    nenvs = 1
    #nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                                nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                max_grad_norm=max_grad_norm, deterministic=deterministic)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if restore_path is not None:
        model.restore(restore_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        ids, obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []

        # Do not train or log if in deterministic mode:
        if deterministic:
            continue

        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                #np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    mblossvals.append(model.train(lrnow, cliprangenow, ids[mbinds], [obs[i] for i in mbinds], returns[mbinds],
                                      masks[mbinds], actions[mbinds], values[mbinds],
                                      neglogpacs[mbinds]))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                #np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow,
                                      [obs[i] for i in mbinds], returns[mbflatinds], masks[mbflatinds], actions[mbflatinds],
                                      values[mbflatinds], neglogpacs[mbflatinds], mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            #ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            #logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
            board_logger.log_scalar("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]), update)
            board_logger.flush()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            if not os.path.isdir(checkdir):
                os.makedirs(checkdir)
            savepath = osp.join(checkdir, "r"+"{:.2f}".format(safemean([epinfo['r'] for epinfo in epinfobuf])) + '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
