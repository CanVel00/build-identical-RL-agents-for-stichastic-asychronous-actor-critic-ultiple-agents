MIN_BATCH = 5
LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
LEARNING_RATE = 5e-3
RMSPropDecaly = 0.99

# Params of advantage (Bellman equation)
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

TRAIN_WORKERS = 10  # Thread number of learning.
TEST_WORKER = 1  # Thread number of testing (default 1)
MAX_STEPS = 20  # Maximum step number.
MAX_TRAIN_NUM = 5000  # Learning number of each thread.
Tmax = 5  # Updating step period of each thread.

# Params of epsilon greedy
EPS_START = 0.5
EPS_END = 0.0


# ParameterServer
class ParameterServer:
    def __init__(self):
        # Identify by name to weights by the thread name (Name Space).
        with tf.variable_scope("parameter_server"):
            # Define neural network.
            self.model = self._build_model()

        # Declare server params.
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
        # Define optimizer.
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSPropDecaly)

    # Define neural network.
    def _build_model(self):
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense1 = Dense(50, activation='relu')(l_input)
        l_dense2 = Dense(100, activation='relu')(l_dense1)
        l_dense3 = Dense(200, activation='relu')(l_dense2)
        l_dense4 = Dense(400, activation='relu')(l_dense3)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense4)
        out_value = Dense(1, activation='linear')(l_dense4)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        return model


# LocalBrain
class LocalBrain:
    def __init__(self, name, parameter_server):
        self.util = Utilty()
        with tf.name_scope(name):
            # s, a, r, s', s' terminal mask
            self.train_queue = [[], [], [], [], []]
            K.set_session(SESS)

            # Define neural network.
            self.model = self._build_model()
            # Define learning method.
            self._build_graph(name, parameter_server)

    # Define neural network.
    def _build_model(self):
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense1 = Dense(50, activation='relu')(l_input)
        l_dense2 = Dense(100, activation='relu')(l_dense1)
        l_dense3 = Dense(200, activation='relu')(l_dense2)
        l_dense4 = Dense(400, activation='relu')(l_dense3)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense4)
        out_value = Dense(1, activation='linear')(l_dense4)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        # Have to initialize before threading
        model._make_predict_function()
        return model

    # Define learning method by TensorFlow.
    def _build_graph(self, name, parameter_server):
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES))
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        # Not immediate, but discounted n step reward
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))

        p, v = self.model(self.s_t)

        # Define loss function.
        log_prob = tf.log(tf.reduce_sum(p * self.a_t, axis=1, keepdims=True) + 1e-10)
        advantage = self.r_t - v
        loss_policy = - log_prob * tf.stop_gradient(advantage)
        # Minimize value error
        loss_value = LOSS_V * tf.square(advantage)
        # Maximize entropy (regularization)
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keepdims=True)
        self.loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        # Define weight.
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        # Define grads.
        self.grads = tf.gradients(self.loss_total, self.weights_params)

        # Define updating weight of ParameterServe
        self.update_global_weight_params = \
            parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

        # Define copying weight of ParameterServer to LocalBrain.
        self.pull_global_weight_params = [l_p.assign(g_p)
                                          for l_p, g_p in zip(self.weights_params, parameter_server.weights_params)]

        # Define copying weight of LocalBrain to ParameterServer.
        self.push_local_weight_params = [g_p.assign(l_p)
                                         for g_p, l_p in zip(parameter_server.weights_params, self.weights_params)]

    # Pull ParameterServer weight to local thread.
    def pull_parameter_server(self):
        SESS.run(self.pull_global_weight_params)

    # Push local thread weight to ParameterServer.
    def push_parameter_server(self):
        SESS.run(self.push_local_weight_params)

    # Updating weight using grads of LocalBrain (learning).
    def update_parameter_server(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            return

        self.util.print_message(NOTE, 'Update LocalBrain weight to ParameterServer.')
        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [[], [], [], [], []]
        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)
        _, v = self.model.predict(s_)

        # Set v to 0 where s_ is terminal state
        r = r + GAMMA_N * v * s_mask
        feed_dict = {self.s_t: s, self.a_t: a, self.r_t: r}  # data of updating weight.
        SESS.run(self.update_global_weight_params, feed_dict)  # Update ParameterServer weight.

    # Return probability of action usin state (s).
    def predict_p(self, s):
        p, v = self.model.predict(s)
        return p

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)


# Agent
class Agent:
    def __init__(self, name, parameter_server):
        self.brain = LocalBrain(name, parameter_server)
        self.memory = []  # Memory of s,a,r,s_
        self.R = 0.  # Time discounted total reward.

    def act(self, s, available_action_list, eps_steps):
        # Decide action using epsilon greedy.
        if frames >= eps_steps:
            eps = EPS_END
        else:
            # Linearly interpolate
            eps = EPS_START + frames * (EPS_END - EPS_START) / eps_steps

        if random.random() < eps:
            # Randomly select action.
            if len(available_action_list) != 0:
                return available_action_list[random.randint(0, len(available_action_list) - 1)], None, None
            else:
                return 'no payload', None, None
        else:
            # Select action according to probability p[0] (greedy).
            s = np.array([s])
            p = self.brain.predict_p(s)
            if len(available_action_list) != 0:
                prob = []
                for action in available_action_list:
                    prob.append([action, p[0][action]])
                prob.sort(key=lambda s: -s[1])
                return prob[0][0], prob[0][1], prob
            else:
                return 'no payload', p[0][len(p[0]) - 1], None

    # Push s,a,r,s considering advantage to LocalBrain.
    def advantage_push_local_brain(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_

        # Create a_cats (one-hot encoding)
        a_cats = np.zeros(NUM_ACTIONS)
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # Calculate R using previous time discounted total reward.
        self.R = (self.R + r * GAMMA_N) / GAMMA

        # Input experience considering advantage to LocalBrain.
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


# Environment.
class Environment:
    total_reward_vec = np.zeros(10)
    count_trial_each_thread = 0

    def __init__(self, name, thread_type, parameter_server, rhost):
        self.name = name
        self.thread_type = thread_type
        self.env = Metasploit(rhost)
        self.agent = Agent(name, parameter_server)
        self.util = Utilty()

    def run(self, exploit_tree, target_tree):
        self.agent.brain.pull_parameter_server()  # Copy ParameterSever weight to LocalBrain
        global frames              # Total number of trial in total session.
        global isFinish            # Finishing of learning/testing flag.
        global exploit_count       # Number of successful exploitation.
        global post_exploit_count  # Number of successful post-exploitation.
        global plot_count          # Exploitation count list for plot.
        global plot_pcount         # Post-exploit count list for plot.

        if self.thread_type == 'test':
            # Execute exploitation.
            self.util.print_message(NOTE, 'Execute exploitation.')
            session_list = []
            for port_num in com_port_list:
                execute_list = []
                target_info = {}
                module_list = target_tree[port_num]['exploit']
                for exploit in module_list:
                    target_list = exploit_tree[exploit[8:]]['target_list']
                    for target in target_list:
                        skip_flag, s, payload_list, target_info = self.env.get_state(exploit_tree,
                                                                                     target_tree,
                                                                                     port_num,
                                                                                     exploit,
                                                                                     target)
                        if skip_flag is False:
                            # Get available payload index.
                            available_actions = self.env.get_available_actions(payload_list)

                            # Decide action using epsilon greedy.
                            frames = self.env.eps_steps
                            _, _, p_list = self.agent.act(s, available_actions, self.env.eps_steps)
                            # Append all payload probabilities.
                            if p_list is not None:
                                for prob in p_list:
                                    execute_list.append([prob[1], exploit, target, prob[0], target_info])
                        else:
                            continue


WorkerThread
class Worker_thread:
    def __init__(self, thread_name, thread_type, parameter_server, rhost):
        self.environment = Environment(thread_name, thread_type, parameter_server, rhost)
        self.thread_name = thread_name
        self.thread_type = thread_type
        self.util = Utilty()

    # Execute learning or testing.
    def run(self, exploit_tree, target_tree, saver=None, train_path=None):
        self.util.print_message(NOTE, 'Executing start: {}'.format(self.thread_name))
        while True:
            if self.thread_type == 'learning':
                # Execute learning thread.
                self.environment.run(exploit_tree, target_tree)

                # Stop learning thread.
                if isFinish:
                    self.util.print_message(OK, 'Finish train: {}'.format(self.thread_name))
                    time.sleep(3.0)

                    # Finally save learned weights.
                    self.util.print_message(OK, 'Save learned data: {}'.format(self.thread_name))
                    saver.save(SESS, train_path)

                    # Disconnection RPC Server.
                    self.environment.env.client.termination(self.environment.env.client.console_id)

                    if self.thread_name == 'local_thread1':
                        # Create plot.
                        df_plot = pd.DataFrame({'exploitation': plot_count,
                                                'post-exploitation': plot_pcount})
                        df_plot.to_csv(os.path.join(self.environment.env.data_path, 'experiment.csv'))
                        # df_plot.plot(kind='line', title='Training result.', legend=True)
                        # plt.savefig(self.environment.env.plot_file)
                        # plt.close('all')

                        # Create report.
                        report = CreateReport()
                        report.create_report('train', pd.to_datetime(self.environment.env.scan_start_time))
                    break
            else:
                # Execute testing thread.
                self.environment.run(exploit_tree, target_tree)

                # Stop testing thread.
                if isFinish:
                    self.util.print_message(OK, 'Finish test.')
                    time.sleep(3.0)

                    # Disconnection RPC Server.
                    self.environment.env.client.termination(self.environment.env.client.console_id)

                    # Create report.
                    report = CreateReport()
                    report.create_report('test', pd.to_datetime(self.environment.env.scan_start_time))
                    break
