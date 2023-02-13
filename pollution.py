import numpy as np
import random
import math
import networkx as nx

def points_within_range(position,radius,size=1):
    """points within range of position at a distance <= radius,
        on the double-torus of period size x size
            position - (int,int) tuple
            radius - float
            size - int
            """
    return [(x%size,y%size) for x in range(position[0]-radius,
                                    position[0]+radius+1)
            for y in range(position[1]-radius,
                            position[1]+radius+1)
            if np.sqrt((x-position[0])**2+(y-position[1])**2) <= radius]

class Agent:
    """
    Agent object with the following methods:
        self.pollute() : add/remove pollution from the world pollution grid
        self.observe() : return pollution at self.position
        self.migrate() : migrating to a a location with least
                            amount of pollution within radius
                            self.migration
        self.imitate() : imitating the strategy of an agent who's site
                            experiences the lowest expense,
                            within radius self.imitate_radius
        self.calc_expense() : calculates and sets self.expense
    """
    def __init__(self,position=(0,0),type='c',R = 5, M=2,M_nu=2,label=1,phi=5,epsilon=0):
        self.label = label # must be a positive number > 0
        self.position = position # position = (x,y) as a tuple of integers
        self.type = type.lower() # type = 'c' or 'd'
        self.migration = M # migratory distance
        self.imitate_radius = min(M_nu,M) # capped the imitate radius to M
        self.phi = phi # cleaning rate of a cooperator
        self.expense = 0 # initially 0 expense
        self.mutation = epsilon # float in [0,1]
        self.radius=R # float

    def pollute(self,world):
        """ self.pollute() : add/remove pollution from the world pollution grid """
        if self.type=='c':
            affected_pts = points_within_range(self.position,1,world.size)
            for pt in affected_pts:
                world.pollution_grid[pt] -= self.phi
        elif self.type=='d':
            affected_pts = points_within_range(self.position,self.radius,world.size)
            for pt in affected_pts:
                x,y=pt
                r = np.sqrt((x-self.position[0])**2+(y-self.position[1])**2)
                if r <= 1:
                    world.pollution_grid[pt] += 1
                elif r <= self.radius:
                    world.pollution_grid[pt] += 1/r**2
        else:
            print("self.type error, must be c or d. Will pollute 0 everywhere")
        # return pollution_grid

    def observe(self,world):
        """ self.observe() : return pollution at self.position """
        return world.pollution_grid[self.position]

    def migrate(self,world):
        """self.migrate() : migrating to a a location with least
                            amount of pollution within radius
                            self.migration"""

        # candiate_pts are sites in range, that are empty and are on the torus
        candidate_pts = [pt for pt in points_within_range(self.position,
                                                            self.migration,
                                                            world.size)
                            if world.lattice_sites[pt]==0]
        if len(candidate_pts)>0:
            # pollution of all candidate sites
            candidate_pol = [world.pollution_grid[candidate_pt]
                                for candidate_pt in candidate_pts]
            min_pol = min(candidate_pol)
            if self.observe(world) > min_pol:
                # restricting candidate_pts to those with the smallest pollution
                candidate_pts = [candidate_pts[i] for i,p in enumerate(candidate_pol)
                                    if p == min_pol]
                world.lattice_sites[self.position] = 0 # emptying world site
                self.position = random.choice(candidate_pts) # for multiple minima
                world.lattice_sites[self.position] = self.label # moving to another world site

    def imitate(self,world):
        """imitating the strategy of an agent who's site
            experiences the lowest expense,
            within radius self.imitate_radius"""

        # Mutation with probability epsilon to flip strategies
        if np.random.uniform()<self.mutation:
            if self.type=='c':
                self.type='d'
            else:
                self.type='c'

        # Imitate the neighbour with minimum expense
        else:
            neighbours = [world.return_agent(world.lattice_sites[pt])[0] for pt in
                                points_within_range(self.position,
                                            self.imitate_radius,
                                            world.size)
                                if world.lattice_sites[pt]>0]
            neighbour_pollution = {a.label:a.expense for a in neighbours}
            best_neighbour = min(neighbour_pollution,key=neighbour_pollution.get)
            self.type = world.return_agent(best_neighbour)[0].type

    def calc_expense(self,world):
        if self.type == 'c':
            self.expense = self.observe(world) + world.f
        elif self.type =='d':
            self.expense = self.observe(world) - world.g
        else:
            self.expense = self.observe(world)



class World:
    """
    World object with the following methods:
        self.populate() : fill self.lattice_sites with agents with parameters
                            (either N with D defectors, or from the list agents)
        self.step() : progress the world by one timestep -
                        1. All agents update strategies
                        2. All agents migrate
                        (3.) self.pollution_grid is reset
                        4. All agents pollute
                        5. All agents calculate expense
        self.pollute() : all agents pollute
        self.calc_expense() : all agents calculate calculate expense
        self.migrate() : all agents migrate
        self.imitate() : all agents update their strategies
        self.spatial_avg() : return spatial average of pollution
                                (ie mean over all lattice sites)
        self.per_capita_pollution() : return per-capita POLLUTION
                                        (ie mean over all occupied sites)
        self.cleaner_rate() : return fraction of cooperators (C/N) in the city
        self.per_capita_expense() : return per-capita EXPENSE
        self.neighbour_list() : return an edge list of all neighbouring pairs
        self.observe_clusters() : returns the list of clusters,
                                    via NetworkX connected_components
        self.return_agent(label) : return the agent object that matches the label
    """
    def __init__(self,L=50,N=10,D=1,agents=[],R=5,M=2,phi=5,M_nu=2,f=1,g=2,epsilon=0):
        self.size = L
        self.pollution_grid = np.zeros([L,L],dtype=np.float64) # the pollution grid / space
        self.lattice_sites = np.zeros([L,L]) # 0 or a label
        self.f=f
        self.g=g
        self.mutation = epsilon
        self.populate(N=N,D=D,agents=agents,R=R,M=M,phi=phi,M_nu=M_nu)
        self.pollute()
        self.calc_expense()

    def populate(self,N=10,D=1,agents=[],R=5,M=2,phi=5,M_nu=1,epsilon=0):
        """ self.populate() : fill self.lattice_sites with agents
                agents is a list of Agent objects, which will supersed N and D
        """
        if len(agents) > 0:
            for a in agents:
                self.lattice_sites[a.position] = a.label
            self.agents = agents
        else:
            empty_sites = [tuple(item) for item in np.argwhere(self.lattice_sites==0).tolist()]
            to_be_inhabited = random.sample(empty_sites,N)
            agents = [Agent(position=to_be_inhabited[i],
                            type='d',label=i+1,R=R,M=M,phi=phi,M_nu=M_nu,epsilon=epsilon)
                            for i in range(D)]
            agents += [Agent(position=to_be_inhabited[i],
                                type='c',label=i+1,R=R,M=M,phi=phi,M_nu=M_nu,epsilon=epsilon)
                                for i in range(D,N)]
            for a in agents:
                self.lattice_sites[a.position] = a.label
            self.agents=agents

    def step(self):
        """
        self.step() : progress the world by one timestep -
                        1. All agents update strategies
                        2. All agents migrate
                        3. All agents pollute
                        4. All agents calculate expense
        """
        self.imitate()
        self.migrate()
        self.pollute()
        self.calc_expense()

    def pollute(self):
        """ self.pollute() : all agents pollute """
        self.pollution_grid=np.zeros([self.size,self.size]) # resets every time step
        for a in self.agents:
            a.pollute(self)

    def calc_expense(self):
        """ self.calc_expense() : all agents calculate calculate expense """
        for a in self.agents:
            a.calc_expense(self)

    def migrate(self):
        """ self.migrate() : all agents migrate """
        for a in self.agents:
            a.migrate(self)

    def imitate(self):
        """ self.imitate() : all agents update strategies"""
        for a in self.agents:
            a.imitate(self)

    def spatial_avg(self):
        """ self.spatial_avg() : return spatial average of pollution
                                (ie mean over all lattice sites) """
        return np.mean(self.pollution_grid)

    def per_capita_pollution(self):
        """ self.per_capita_pollution() : return per-capita POLLUTION """
        return np.mean([self.pollution_grid[a.position] for a in self.agents])

    def cleaner_rate(self):
        """ self.cleaner_rate() : return fraction of cooperators (C/N) in the city """
        return len([a for a in self.agents if a.type=='c'])/len(self.agents)

    def per_capita_expense(self):
        """ self.per_capita_expense() : return per-capita EXPENSE """
        return np.mean([a.expense for a in self.agents])

    def neighbour_list(self):
        """ self.neighbour_list() : return an edge list of all neighbouring pairs """
        neigh_list = []
        for a in self.agents:
            pos=a.position
            neigh_sites = [((pos[0]+1)%self.size,pos[1]),(pos[0],(pos[1]+1)%self.size),
                            ((pos[0]-1)%self.size,pos[1]),(pos[0],(pos[1]-1)%self.size)]
            neigh_agents = np.array([self.lattice_sites[p] for p in neigh_sites],dtype=int)
            for b in neigh_agents[neigh_agents!=0]:
                neigh_list += [(a.label,b)]
        return neigh_list

    def observe_clusters(self):
        """ self.observe_clusters() : returns the list of clusters,
                                        via NetworkX connected_components """
        # Convert the grid into a NetworkX network first
            # then output all connected components
        return [c for c in nx.connected_components(nx.from_edgelist(self.neighbour_list()))]

    def return_agent(self,label):
        """ self.return_agent(label) : return the agent object that matches the label """
        return [a for a in self.agents if a.label==label]
