import utils
import numpy as np

class TSPEnv():

    def __init__(self):
        super(TSPEnv, self).__init__()

        self.observation_space = None

    def reset(self, points, tour, T=None):

        self.T = T
        self.points = points
        self.state = np.copy(self.points)

        self.current_step = 0

        #init memory
        self.hist_best_distances =  []
        self.hist_current_distances = []

        self.tour = tour

        self.reset_tour = self.tour.copy()

        #distances: list of lists with all distances for points
        self.distances = utils.calculate_distances(self.state)
        self.distances = np.rint(self.distances*10000)
        self.distances = self.distances.astype(int)

        #state : reorder the points with the random tour before starting
        # this is the initial state
        self.state = self.state[self.tour, :]
        self.best_state = np.copy(self.state)

        # keep_tours : tour for computing distances (invariant to state)
        self.keep_tour = self.tour.copy()

        # tour_distance : distance of the current tour
        self.tour_distance = utils.route_distance(self.keep_tour,
                                                  self.tour_distance)

        # current best : save the initial tour (keep_tour) and distance

        self.current_best_distance = self.tour_distance
        self.current_best_tour = self.keep_tour.copy()

        #update memory
        self.hist_best_distances.append(self.current_best_distance)
        self.hist_current_distances.append(self.tour_distance)

        return self._next_observation(), self.best_state

    def _next_observation(self):
        """

        :param self:
        :return: next observation of the tsp environment
         """
        observation = self.state
        return observation

    def step(self, action):
        self.current_step += 1

        reward = self._take_action(action)
        observation = self._next_observation()
        done = False
        if self.T is not None:
            self.T -= 1

        return observation, reward, done, self.best_state


    def _take_action(self, action):
        """
        take action in the tsp env
        :param action:
        :return:
        """
        self.tour = utils.swap_2opt(self.tour,
                                    action[0],
                                    action[1])

        self.new_keep_tour, self.new_tour_distance = utils.swap_2opt_new(self.keep_tour,
                                                                         action[0],
                                                                         action[1],
                                                                         self.tour_distance,
                                                                         self.distances)
        self.state = self.state[self.tour, : ]
        self.tour_distance = self.new_tour_distance.copy()
        #if (self.current_best_distane > self.tour_distance):


