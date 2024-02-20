import tensorflow as tf
import rlcard
import os

from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger

# Make environment
env = rlcard.make('no-limit-holdem')
eval_env = rlcard.make('no-limit-holdem')

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
save_plot_every = 1000
evaluate_num = 1000
episode_num = 10000

# Set the the number of steps for collecting normalization statistics
# and intial memory size
memory_init_size = 1000
norm_step = 100

# The main paths and folders for saving all the logs and learning and loss curves
root_path = './Experiment/Results/'

#Loss result paths and folder
loss_path = root_path + 'Loss/'
loss_log_path = loss_path + 'loss_log.txt'
loss_csv_path = loss_path + 'loss_performance.csv'
loss_figure_path = loss_path + 'Loss_Figures/'

#Reward paths and folder
reward_path = root_path + 'Rewards/'
reward_log_path = reward_path + 'reward_log.txt'
reward_csv_path = reward_path + 'reward_performance.csv'
reward_figure_path = reward_path + 'Reward_Figures/'

#Model path for saving model
model_saved_path = './Model/'

# Set a global seed
set_global_seed(0)

with tf.compat.v1.Session() as sess:

    # Create and set the agents
    global_step = tf.Variable(0, name='global_step', trainable=False)

    agent = DQNAgent(sess,
                     scope='dqn',
                     action_num=env.action_num,
                     replay_memory_size=1000000,
                     replay_memory_init_size=memory_init_size,
                     update_target_estimator_every=1000,
                     epsilon_start=1.0,
                     epsilon_end=0.1,
                     epsilon_decay_steps=20000,
                     norm_step=norm_step,
                     state_shape=env.state_shape,
                     mlp_layers=[512, 512],
                     batch_size=512,
                     learning_rate = 0.0005,
                     discount_factor = 0.95
                     )

    random_agent = RandomAgent(action_num=eval_env.action_num)

    nfsp_agent = NFSPAgent(sess,
                           scope = 'nfsp',
                           action_num=env.action_num,
                           state_shape=env.state_shape,
                           hidden_layers_sizes=[512,512],
                           q_replay_memory_size=int(1e5),
                           q_replay_memory_init_size=memory_init_size,
                           q_mlp_layers=[512, 512]
                           )

    saver = tf.train.Saver()

    #Create the model saved path and folder or load the saved model
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
        print("File directory is created.")
    else:
        saver.restore(agent.sess, tf.train.latest_checkpoint(model_saved_path))
        print("\n\nModel restored.")

    sess.run(tf.global_variables_initializer())

    env.set_agents([agent, random_agent,nfsp_agent])
    eval_env.set_agents([agent, random_agent, nfsp_agent])

    # Count the number of steps
    step_counter = 0

    # Set logger to plot the learning an loss curve
    Reward = Logger(xlabel = 'Timestep', ylabel = 'Reward', legend='Average Rewards in Each Timestep', log_path = reward_log_path, csv_path = reward_csv_path)
    Loss = Logger(xlabel = 'Step', ylabel = 'Loss', legend = 'The Loss in each Step', log_path = loss_log_path, csv_path = loss_csv_path  )


    for episode in range(episode_num):
        

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)
            step_counter += 1

            # Train the agent
            train_count = step_counter - (memory_init_size + norm_step)
            if train_count > 0:

                loss = agent.train()

                Loss.log('Step: {}, loss: {}'.format(step_counter, loss))

                # Add point to logger
                Loss.add_point(x = step_counter, y = loss )
    
        # Evaluate the performance. Play with random and NFSP agents.
        if episode % evaluate_every == 0:
            reward = 0  
            for eval_episode in range(evaluate_num):
                _, payoffs = eval_env.run(is_training=False)

                reward += payoffs[0]

            Reward.log('\n########## Evaluation ##########')
            Reward.log('Timestep: {} Average Reward: {}'.format(env.timestep, float(reward)/evaluate_num))

            # Add point to logger
            Reward.add_point(x=env.timestep, y=float(reward)/evaluate_num)

        # Make plot
        if episode % save_plot_every == 0 and episode > 0:
            Reward.make_plot(save_path = reward_figure_path+str(episode)+'.png')
            Loss.make_plot(save_path = loss_figure_path+str(episode)+'png' )

    # Make the final plot
    Reward.make_plot(save_path = reward_figure_path+str(episode)+'.png')
    Loss.make_plot(save_path = loss_figure_path+str(episode)+'png' )

    # Save the model
    saver.save(agent.sess, os.path.join(model_saved_path, 'model'))
    print("\n\nModel saved in path: ", model_saved_path)
    
    



