import numpy as np
from tqdm import tqdm
import random
import math
import time
import pandas as pd 
from functools import reduce
import seaborn as sns
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats
from csv import reader

# If False, then the plots are showed on screen, if true they are saved in .pgf format
export_plots = False

# These are the global settings for the simulation. Which corresponds to the parameters describes in the dissertation. Beware that a sligthly different terminology is used 
# here then in the actual dissertation. For more info see the readme file. 
SETTINGS = {
    # The maximum amount of sugar at any point on the graph (setting this to "a" means the the sugar potential is distributed U[0,a]) 
    'Maximum sugar' : 10,
    # The rate at which the sugar regrows
    'Sugar regrwoth rate' : 0.8,
    # How far the individuals see in each direction 
    'Individual sight' : 2, 
    # How much sugar the individuals have eat each turn to stay alive
    'Metabolism' : 2, 
    # The proportion of individuls that attacks other individuals (if the find that to be "profitable")
    'Proportion of agressors' : 0.3, 
    # For how many rounds the simulations is run
    'Number of rounds' : 1000, 
    # The size of the graph
    'Size of lattice' : 30,
    # The number of individuals that are initiated on the graph 
    'Number of individuals' : 700,
    # Bias in favour of attacker. "Gamma" in the dissertation
    'Bias in favour of attacker' : 2.5,
    # How the strength is distributed among the individuals
    'Strength distribution' : {'Family' : 'Uniform', 
                                'Parameter 1' : 5,
                                'Parameter 2' : 5,
                                }, # Alternativs: Gamma, Exponential, Unifrom. In case of Exponential only Parameter 1 should be given
    # This is not working at the moment
    'Decions mechanism for joining coalitions' : 'Intricate',
    # This is not working at the moment
    'Sugar allocation' : 'Random',
    # How much sugar each agent have when the simulation starts
    'Starting sugar level' : 6,
    # If True, the attacked individuals/coalition takes the sugar from the attacking individual
    'Defenders seek retribution' : False,
    # This is not working at the moment 
    'Move with your coalition' : False,
    # This is how much sugar it cost to have a conflict regardless of whether or not you win or lose. Not used in dissertation
    'Cost of combat' : 0,
    # This is the voting mechanism used by the coalitions to decided on membership of potential new members
    'Voting Mechanism' : 'Majority', # Veto or Majority
    # This decides which attractiveness function is used. High is the one used in treatments 1-5, Low is the one used in base treatment
    'Returns to Scale' : 'High', # High or Low
}

if export_plots == True:  
    plt.rcParams.update({'font.size': 8})
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

# read csv file as a list of lists
# Importing a list of names to name each agent, I found this made the de-bugging a lot easier. 
with open('names.csv') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

names = []
for r in list_of_rows:
    names.append(r[0])

# For producing diagrams showing the different coalitions.
markers = ['.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','D','d','|','_',0,1,2,3,4,5,6,7,8,9,10,11]

class Individual: 
    '''A indiviaul living on the lattice''' 
    def __init__(self, sugar, sight, metabolism, location, name, agressor_type, strength, prop_hawks, bias_in_favour_of_attacker): 
        self.sugar = sugar # Gives the individual a strating amount of sugar 
        self.location = location # Gives the individual a location, the initial location has to be defined in advanced
        self.name = name # A name for each individual, so that they can be easily identified, also for fun. 
        self.alive = True # Signifies wheather or not the individual is alive or not
        self.sight = sight # How long in each direction an individual can see
        self.metabolism = metabolism # How much the agents eat each round
        self.type = agressor_type # If they are an agressor or not
        self.strength = strength # What strenght the individuals posses
        self.coalition = ProperCoalition([self]) # Which coalition the agents belong to. 
        self.prop_hawks = prop_hawks # What the individual thinks about the propbability is the
        self.bias_in_favour_of_attacker = bias_in_favour_of_attacker # Just so that each individual knows what the bias is. 
    
    def __repr__(self):
        return self.name
    
    def look_for_sugar(self, lattice):
        ''' creates a dictionary of the nodes visible to the individual and the amount of sugar 
        availiable a that node
        arguments: lattice - the lattice
        returns: a sorted dictionary with respect to sugar amount, from lowest to highest
        '''
        N = lattice.size
        sugar_dict = dict()
        x0, y0 = self.location 

        for i in range(-self.sight, self.sight + 1): 
            for j in range(-self.sight, self.sight + 1):
                if 0 <= x0 + i < N and 0 <= y0 + j < N: 
                    spot = (x0 +i, y0 + j) 
                    sugar_dict[spot] = lattice.sugar[spot]

        sorted_sugar_dict = {k: v for k, v in sorted(sugar_dict.items(), key=lambda item: item[1], reverse=True)} # I sort the list for highest amount of sugar to lowest, of understandable reasons            
        return sorted_sugar_dict
                    
    def move(self, individuals, lattice):
        '''Moves the individual to a random unoccupied spot on the latice
        Arguments:  individauls - a dictionary of individuals
        ''' 
        sorted_sugar_dict = self.look_for_sugar(lattice)
        
        # Looking thorugh to locations on the list to find the one with the most sugar that are also unoccupied and moves there
        new_location_found = False
        new_location = self.location
        iteration = 0
        while new_location_found == False and iteration < len(sorted_sugar_dict): 
            x, y = list(sorted_sugar_dict.keys())[iteration]
            x_y_unoccupied = True 
            for k,v in individuals.items(): 
                if v.location == (x,y) and k != self.name and v.alive==True:
                    x_y_unoccupied = False
                    break
            if x_y_unoccupied: 
                new_location_found = True
                new_location = (x,y)
            iteration += 1 
            # print(str(self.name) + ' iteration: ' + str(iteration)) 
        self.location = new_location

    def move_hawk(self, lattice, individuals, nbhd, cost_of_combat, bias_in_favour_of_attacker, combat_log, round_number, SETTINGS): 
        '''
        This function moves the hawks of the simulation. I.e. finds the most attractive individaul close to the hawks and attacks that individuals
        if that individuals/coalitions attractiveness is lower than the maximum sugar in the sight of the self. 
        
        Inputs: lattice - where the simulatio takes place, individuals - a dictionary of individuals, nbhd - a list of the individuals in the self's neighbouthood
                individuals - a dictionary of individuals
                nbhd - a list of individuals that are in the neighbourhood of the self
                cost_of_combat - the cost of combat
                bias_in_favour_of_attacker  - the bias in favour of the attacker (specified in SETTINGS)
                combat_log - a dictionary where all the combat information is stored (i.e. who attacked whom, who won and so o)
                round_number - which round number is the current one (needed for the combat log)
                SETTINGS - the dicitionary with the settings for the simulation
        '''

        max_attractiveness_in_nbhd = 0
        most_attractive_individual = []
        # You should not attack yourself! That would be stupid.
        nbhd_copy = nbhd.copy()
        nbhd_copy.remove(self)
        for i in nbhd_copy: 
            i_attractiveness = attractiveness_function(self.strength, self.find_strength_of_coalition(i.coalition, individuals, nbhd), self.find_sugar_of_coalition(i.coalition, individuals, nbhd), len(self.find_members_of_coalition_in_nbhd(i.coalition, individuals, nbhd)), self.bias_in_favour_of_attacker)
            if i_attractiveness > max_attractiveness_in_nbhd:
                max_attractiveness_in_nbhd = i_attractiveness
                if len(most_attractive_individual) > 0: 
                    most_attractive_individual.pop()
                    most_attractive_individual.append(i)
                else: 
                    most_attractive_individual.append(i)

        sugar_dict = self.look_for_sugar(lattice)
        max_sugar = list(sugar_dict.items())[0][1]

        if max_attractiveness_in_nbhd - cost_of_combat - max_sugar > 0:
            combat_coalition(self, most_attractive_individual[0], most_attractive_individual[0].coalition, cost_of_combat, individuals, bias_in_favour_of_attacker, combat_log, round_number, SETTINGS)    
        else:  
            self.move(individuals, lattice)
            
    def find_nbhd(self, individuals):
        '''
        This function takes in a particular individual and a dictionary of individual and returns a list of all other individuals in the neighbourhood
        of that first individual.

        inputs: individual - an individual , 
                individuals - a dicitionary of individuals 

        returns: a list of individuals each in the neighbourhood of the first individual 
        '''
        sight = self.sight # defines the size of the neighbourhood 
        x0, y0 = self.location # featching the location of the individual

        nbhd = [] 
        individuals_in_nbhd = []
        # The following loop creates a list of a points in the neighbourhood of the individual
        for n in range(-sight ,sight):
            for k in range(-sight, sight):
                xn = x0 + n
                yk = y0 + k 
                nbhd.append((xn, yk))
                # Seems unnecessary the check wheather a location is actually on the lattice, since no individual will have that Location any way

        # The following loop goes thorugh all individuals and checks in there location is in the neighbourhood list (created in the previous loop) of the individuals
        for k,v in individuals.items():
            if v.location in nbhd: 
                individuals_in_nbhd.append(v)
        
        return individuals_in_nbhd
    
    def harvest(self, lattice):
        ''' Takes in an individual and a lattice and harvest the sugar for the node on the lattice that represents the individuals location
        i.e. adds sugar to the individuals sugar amount and takes it away from the node

        inputs: 
            lattice - the "lattice" where it all takes place
        '''
        self.sugar += lattice.sugar[(self.location)]
        lattice.sugar[(self.location)] = 0
    

    def eat(self): 
        "Takes the individuals and removes the amount of sugar defined in his metabolism"
        self.sugar += -self.metabolism
        if self.sugar < 0:
            self.alive = False
            self.coalition.members.remove(self)

    def assess_probability_of_attack_individual(self, strength, sugar, individuals, nbhd, size_of_coal):
        '''
        This function find the probability of an individual of being attack (according to the information that the individuals have) 

        inputs:
                individuals - a dictionary whose values are the rest of the individuals in the
                simulation 
                strenght - the individual's strength (the individual or his coalition that is)
                sugar the individual's strength (or the individual's coalition)
                nbhd - a list of individuals in the neighbourhood of the individual
                size_of_coal - the size (number of members) of the coalition of the individuals coalition 

        
        returns: probability of attack and defeat
        '''
        coalitions_in_nbhd = self.find_coalitions_in_nbhd(individuals, nbhd)
        num_of_coal =  len(coalitions_in_nbhd)
        if num_of_coal >= 2: 
            number_of_neighbours = len(nbhd) - 1
            probability_hawk = self.prop_hawks #Should the individual know this or should he guess this
            # Begin with calculating how likely the neighbours are individually to have another neighbour that is more attractive to attack then the individual 
            list_of_probabilities = []
            nbhd_without_self = [i for i in nbhd if i != self] 
            bias_in_favour_of_attacker = self.bias_in_favour_of_attacker

            # This loops goes thorugh the individuals in the assessed individuals neighbourhood and calculated the probability of that individual won't attack 
            for i in nbhd_without_self:
                own_attrativeness_wrt_i = attractiveness_function(i.strength, strength, sugar, size_of_coal, self.bias_in_favour_of_attacker) # needed to plug into the cdf of the functions later 
                i_probability_of_win = own_attrativeness_wrt_i / sugar 

                coalition_attractiveness_dict = {}
                for k in coalitions_in_nbhd:
                    coalition_attractiveness_dict[k] = attractiveness_function(i.strength, self.find_strength_of_coalition(k, individuals, nbhd), self.find_sugar_of_coalition(k, individuals, nbhd), len(self.find_members_of_coalition_in_nbhd(k, individuals, nbhd)), self.bias_in_favour_of_attacker)

                sample_mean = sum(map(lambda k: coalition_attractiveness_dict[k], coalitions_in_nbhd))/len(coalitions_in_nbhd)
                sample_variance = sum(map(lambda k: (coalition_attractiveness_dict[k] - sample_mean)**2, coalitions_in_nbhd))/(len(coalitions_in_nbhd) - 1) # calculates the sample variance of attractiveness
                
                if sample_mean != 0 and sample_variance != 0: 
                    k = sample_mean**2/sample_variance # to fit a gamma distribution using the sample mean and variance (using methods of moments)
                    theta = sample_variance/sample_mean # to fit a gamma distribution using the sample mean and variance (using methods of moments)
                    probability_of_my_attractiveness_greatest = (scipy.stats.gamma.cdf(own_attrativeness_wrt_i, k, loc=0, scale=theta))**number_of_neighbours #Survial function based on the gamma distribution, i.e. the probability
                    probability_of_attack_and_defeat_from_i = probability_of_my_attractiveness_greatest*probability_hawk*i_probability_of_win # The probability of i attacking the individuals is the probabilty of him being an agressor and am being the individua in his neighbourhood the is most attractive to attack
                    probability_of_no_defeat_to_i = 1 - probability_of_attack_and_defeat_from_i # one minus the above is the probability of not being attacked by i (the is useful we compounding the probabilities of all individuals inthe neighbourhood)
                    list_of_probabilities.append(probability_of_no_defeat_to_i)
                else:   
                    probability_of_no_defeat_to_i = 1 - probability_hawk*i_probability_of_win
                
            probability_of_no_defeat = 1
            for p in list_of_probabilities: # the loop computes the probability of no agent attacking me (easier to do then the other way around)
                probability_of_no_defeat = probability_of_no_defeat*p
            probability_of_defeat = 1 - probability_of_no_defeat # One minus is the probability of me being attacked. This the probability we are looking for
            return probability_of_defeat
        elif num_of_coal == 2:
            probability_of_defeat_to_i = probability_hawk*i_probability_of_win
            return probability_of_defeat_to_i
        else:
            return 0

    def assess_probability_of_attack_coalition(self, coalition, individuals, nbhd):
        '''
        This function assess the probability of attack (and defeat) for another coalition in the neighbourhood of the seld

        inputs: 
                coalition - the coalition in question
                individuals - a dicitionary of individuals
                nbhd - a list of individuals in the neighbourhood of the self

        returns: 
                a probability of attack and defeat
        '''
        # Find the relevant individuals
        members_in_nbhd = self.find_members_of_coalition_in_nbhd(coalition, individuals, nbhd)
        strenght_of_colition_in_nbhd = self.find_strength_of_coalition(coalition, individuals, nbhd)
        sugar_of_coalition_in_nbhd = self.find_sugar_of_coalition(coalition, individuals, nbhd)

        probability_of_defeat = 1 - (1 - self.assess_probability_of_attack_individual(strenght_of_colition_in_nbhd, sugar_of_coalition_in_nbhd, individuals, nbhd, len(members_in_nbhd)))**len(members_in_nbhd)

        return probability_of_defeat

    def find_members_of_coalition_in_nbhd(self, coalition, individuals, nbhd):
        ''' Finds the members of a coalition in the neighbourhood of the self.

        inputs: 
                coalition - the coalition in question
                individuals - a dictionary of individuals
                nbhd - a list of individuals in the neighbourhood of the self

        returns: 
                a list of individuals
        '''

        members_in_nbhd = []
        for i in coalition.members:
            if i in nbhd: 
                members_in_nbhd.append(i)
        
        return members_in_nbhd

    def find_strength_of_coalition(self, coalition, individuals, nbhd): 
        '''
        Finds the sum of strengths for a given coalition in the neighbourhood of the self

        inputs: 
                coalition - the coalition in question
                individuals - a dictionary of individuals
                nbhd - a list of individuals in the neighbourhood of the self
        
        returns:  
                the sum of strengths

        '''
        members_in_nbhd = self.find_members_of_coalition_in_nbhd(coalition, individuals, nbhd)

        strengths = [i.strength for i in members_in_nbhd]
        strenght_of_colition_in_nbhd = sum(strengths)

        return strenght_of_colition_in_nbhd

    def find_sugar_of_coalition(self, coalition, individuals, nbhd):
        '''
        Finds the sum of sugar for a given coalition in the neighbourhood of the self

        inputs: 
                coalition - the coalition in question
                individuals - a dictionary of individuals
                nbhd - a list of individuals in the neighbourhood of the self
        
        returns:  
                the sum of sugar

        '''
        members_in_nbhd = self.find_members_of_coalition_in_nbhd(coalition, individuals, nbhd)

        sugar = [i.sugar for i in members_in_nbhd]
        sugar_of_colition_in_nbhd = sum(sugar)

        return sugar_of_colition_in_nbhd
    
    def find_coalitions_in_nbhd(self, individuals, nbhd):
        '''Finds all coalitions respresented in the neighbourhood of the self

        inputs: 
                individuals - a dictionary with individuals
                nbhd - a list of individuals in the neighbourhood of the self

        returns:
                a list of coalitions in the neighbourhood of the self
        '''
        coalitions_in_nbhd = []
        for i in nbhd:
            coalitions_in_nbhd.append(i.coalition)
        
        return list(set(coalitions_in_nbhd))

    def compute_value_of_joining_coalition(self, coalition, individuals, nbhd): 
        '''Computes the expected vaule of joining a given coalition as percived by the self

        Inputs:
                coalition - the coalition in question
                individuals - a dicitionary of individuals
                nbhd - a list of individuals in the neighbourhood of the self
        
        Returns: 
                the expected vaule
        '''
        strength = self.find_strength_of_coalition(self.coalition, individuals, nbhd)
        sugar = self.find_sugar_of_coalition(self.coalition, individuals, nbhd)
        
        potential_coalition_members = coalition.members.copy() 
        if self not in coalition.members:
            potential_coalition_members.append(self)
        potential_coalition = ProperCoalition(potential_coalition_members)

        EV = self.sugar*(self.assess_probability_of_attack_coalition(potential_coalition, individuals, nbhd) - self.assess_probability_of_attack_coalition(self.coalition, individuals, nbhd))
        return EV

    def decide_on_coalition_membership(self, individuals, nbhd):
        '''Useing the decision rule specified in the dissertation decide whether join a coalition

        Inputs: 
                individuals - a dictionary of individuals
                nbhd - a list of individuals in the neighbourhood of the self
        '''
        coalitions_in_nbhd = self.find_coalitions_in_nbhd(individuals, nbhd)
        values_of_coalitions = {}

        for c in coalitions_in_nbhd:
            values_of_coalitions[c] = self.compute_value_of_joining_coalition(c, individuals, nbhd)
        
        if len(self.coalition.members) > 1: 
            go_at_it_alone_coalition = ProperCoalition([self])
            values_of_coalitions[go_at_it_alone_coalition] = self.compute_value_of_joining_coalition(go_at_it_alone_coalition, individuals, nbhd) 
    
        best_coalition = self.coalition
        min_value = 0 
        for k,v in values_of_coalitions.items(): 
            if v < min_value:
                min_value = v
                best_coalition = k
        
        if best_coalition.number != self.coalition.number: 
            best_coalition.vote_on_membership(self, individuals)

# Decide what rule the coalition uses for voting on memebership
majority = (SETTINGS['Voting Mechanism'] == 'Majority')

class ProperCoalition: 
    def __init__(self, members):
        self.members = members 
        self.number = random.randint(1,1000000000000000) # so that we can refer to different coalitions good for debugging purposes
    
    def vote_on_membership(self, individual, individuals, majority=majority): 
        '''This takes in an individual, and determines whether that individual can join the coalition or not

        inputs: 
            individual - the individual who wants to join
            individuals - a dicitionary with the individuals in the simulation
            majority - boolean, if true then a majority priciple is applied to aspiring memebers 
        
        '''
        length = len(self.members)
        potential_coalition_members = self.members.copy()
        potential_coalition_members.append(individual)
        potential_coalition = ProperCoalition(potential_coalition_members)

        if majority:
            tally = 0
            for i in self.members:
                nbhd = i.find_nbhd(individuals)
                value_for_i = i.assess_probability_of_attack_coalition(self, individuals, nbhd) - i.assess_probability_of_attack_coalition(potential_coalition, individuals, nbhd)
                if value_for_i > 0: 
                    tally += 1
                if value_for_i == 0:
                    tally += 0.5
            if float(tally) > float(length)/2: 
                self.members.append(individual)
                individual.coalition.members.remove(individual)
                individual.coalition = self
        else:
            veto_exercised = False
            for i in self.members:
                nbhd = i.find_nbhd(individuals)
                if i.assess_probability_of_attack_coalition(self, individuals, nbhd) - i.assess_probability_of_attack_coalition(potential_coalition, individuals, nbhd) < 0:
                    veto_exercised = True
                    break
            if not veto_exercised: 
                self.members.append(individual)
                individual.coalition.members.remove(individual) 
                individual.coalition = self
                
class Lattice: 
    '''The lattice where upon everything takes pace
    Arguments: N - The size of the lattice (we only allow)
    '''

    def __init__(self, SETTINGS):
        N = SETTINGS['Size of lattice']
        max_sugar_level = SETTINGS['Maximum sugar']
        sugar_regrowth_rate = SETTINGS['Sugar regrwoth rate']
        self.size = N
        self.sugar = np.zeros((N,N)) # the actual sugar level for a given point on the graph
        self.max_sugar = np.zeros((N,N))
        self.regrowth_rate = SETTINGS['Sugar regrwoth rate']
        
        for i in range(N): 
            for j in range(N):
                max_sugar = np.random.uniform(0, max_sugar_level)
                self.sugar[(i,j)] = max_sugar
                self.max_sugar[(i,j)] = max_sugar
        
    def regrow_sugar(self):
        for i in range(self.size): 
                for j in range(self.size):
                        with_growth = self.sugar[(i,j)] + self.regrowth_rate*self.max_sugar[(i,j)]
                        self.sugar[(i,j)] = min((with_growth, self.max_sugar[(i,j)])) 

if SETTINGS['Returns to Scale'] == 'High':
    def attractiveness_function(attacker_strength, attacked_coalition_strength, attacked_coalition_sugar, size_of_coal, bias_in_favour_of_attacker): 
        return ((attacker_strength+bias_in_favour_of_attacker)/(attacker_strength + attacked_coalition_strength**size_of_coal + bias_in_favour_of_attacker))*(attacked_coalition_sugar)
if SETTINGS['Returns to Scale'] == 'Low':
    def attractiveness_function(attacker_strength, attacked_coalition_strength, attacked_coalition_sugar, size_of_coal, bias_in_favour_of_attacker): 
        return ((attacker_strength+bias_in_favour_of_attacker)/(attacker_strength + attacked_coalition_strength + bias_in_favour_of_attacker))*(attacked_coalition_sugar)
            
def combat(attacker, attacked, individuals, cost_of_combat, lattice, bias_in_favour_of_attacker, combat_log, round_number, SETTINGS): 
    '''This function excecutes a combat between a attacker and attacked. If the attacker wins the combat he takes the attcked individauls sugar
        and the attacked agent moves to the closes uninhabited location on the lattice

        inputs: attacker - the attacking individual, 
        attacker - the individual being attacked, 
        individuals - a dictionary of the individuals involved in the simulation
        cost_of_combat - the cost of combat (to the attacker)
        lattice - the lattice in question
        bias_in_favour_of_the_attacker - the bias in favour of the attacker
        combat_log - a dictionary that saves the outcomes from the combats in each round, good for debugging
        round_number - the round number
        SETTINGS - a dicitionary with all the global settings

    '''
    
    p = (attacker.strength+bias_in_favour_of_attacker)/(attacked.strength + attacker.strength+bias_in_favour_of_attacker) # The probability of an individual winning a combat is proportional to that individuals strength
  
    outcome_of_combat = scipy.stats.bernoulli.rvs(p, size=1)

    if outcome_of_combat == 1: # This is the case when the attacker is the winner
        attacker.sugar += (attacked.sugar - cost_of_combat)
        attacked.sugar = 0
        attacker.location = attacked.location
        attacked.location = find_closest_unoccupied_loc(attacked, individuals, lattice)
        log_entry = dict(
            Attacker = attacker.name,
            Attacked = attacked.name,
            Winner = attacker.name
        )
        combat_log["round " + str(round_number)].append(log_entry) 
    
    if outcome_of_combat == 0: # This is the case when the attacked is the winner (where nothing happens but the attacker loses the cost of combat)
        attacker.sugar += -cost_of_combat
        log_entry = dict(
            Attacker = attacker.name,
            Attacked = attacked.name,
            Winner = attacked.name
        )
        combat_log["round " + str(round_number)].append(log_entry) 

def combat_coalition(attacker, attacked, coalition, cost_of_combat, individuals, bias_in_favour_of_attacker, combat_log, round_number, SETTINGS): 
    '''This function excecutes a combat between a attacker and attacked - this version with coalitions. If the attacker wins the combat he takes the attcked individauls sugar
        and the attacked agent moves to the closes uninhabited location on the lattice

        inputs: attacker - the attacking individual, 
        coalition - the coalition of the memeber in quesiton
        attacker - the individual being attacked, 
        individuals - a dictionary of the individuals involved in the simulation
        cost_of_combat - the cost of combat (to the attacker)
        lattice - the lattice in question
        bias_in_favour_of_the_attacker - the bias in favour of the attacker
        combat_log - a dictionary that saves the outcomes from the combats in each round, good for debugging
        round_number - the round number
        SETTINGS - a dicitionary with all the global settings

    '''

    individuals_in_nbhd = attacked.find_nbhd(individuals)
    coalition_members_in_nbhd = list(filter(lambda x: x in coalition.members, individuals_in_nbhd))
    size_of_coal = len(coalition_members_in_nbhd)
    sum_of_strengths = sum(map(lambda x: x.strength,coalition_members_in_nbhd))
    if SETTINGS['Returns to Scale'] == 'High':
        p = (attacker.strength + bias_in_favour_of_attacker)/(attacker.strength + sum_of_strengths**size_of_coal + bias_in_favour_of_attacker)
    else: 
        p = (attacker.strength + bias_in_favour_of_attacker)/(attacker.strength + sum_of_strengths + bias_in_favour_of_attacker)
    outcome_of_combat = scipy.stats.bernoulli.rvs(p, size=1)
    retribution = SETTINGS['Defenders seek retribution']

    if outcome_of_combat == 1: # This is the case when the attacker is the winner
        total_sugar = 0
        for i in coalition_members_in_nbhd: # All individuals in the coalition loose their sugar (perhaps this makes attacking too good?)
            total_sugar += i.sugar
            i.sugar = 0
        attacker.sugar += (total_sugar - cost_of_combat)
        attacker.location = attacked.location
        attacked.location = find_closest_unoccupied_loc(attacked, individuals, lattice)
        log_entry = dict(
            Attacker = attacker.name,
            Attacked = attacked.name,
            Winner = attacker.name,
            Coalition = True, 
            CoalitionMembersInvolved = [i.name for i in coalition_members_in_nbhd]
        )
        combat_log["round " + str(round_number)].append(log_entry) 
    
    if outcome_of_combat == 0: # This is the case when the attacked is the winner (where nothing happens but the attacker loses the cost of combat)
        attacker.sugar += -cost_of_combat
        if retribution: 
            sugar_left = attacker.sugar
            n = len(coalition_members_in_nbhd)
            for m in coalition.members: 
                m.sugar += sugar_left/n
        log_entry = dict(
            Attacker = attacker.name,
            Attacked = attacked.name,
            Winner = attacked.name, 
            Coalition = True, 
            CoalitionMembersInvolved = [i.name for i in coalition_members_in_nbhd]
        )
        combat_log["round " + str(round_number)].append(log_entry) 
        
def find_closest_unoccupied_loc(individual, individuals, lattice):
    '''Finds the closest unoccupied location to a given individual

        inputs: 
            individual - the individuals whose closest unoccupied location is to be found
            individuals - a dictionary of all the individuals inthe simulation
            lattice - the lattice in question
    '''
    sight = individual.sight
    size_of_lattice = lattice.size
    x0, y0 = individual.location
    individuals_in_nbhd = individual.find_nbhd(individuals)

    nbhd = [] #adding all locations in ones neighbourhood
    for i in range(-sight, sight + 1): 
        for j in range(-sight, sight + 1):
            if 0 <= x0 + i < size_of_lattice and 0 <= y0 + j < size_of_lattice:
                loc = (x0 + i, x0 + j)
                nbhd.append(loc)
    
    if individual.location in nbhd:
        nbhd.remove(individual.location) # Your not going to be allowed to stay where you are - the purpose of the function is to find the closest unoccupied location the isn't your own
    for i in individuals_in_nbhd:
        if i in nbhd: 
            nbhd.remove(i.location)
    
    # Using the graph metric the decide how close someone is one the lattice... 
    
    nbhd_distances_dict = {}
    for k in nbhd: 
        dist = max(abs(k[0] - x0), abs(k[1] - y0))
        nbhd_distances_dict[k] = dist

    return min(nbhd_distances_dict, key=nbhd_distances_dict.get)

def random_initial_locations(number_of_players, size_of_lattice): 
    '''The purpose of this function is to draw a sample from the possible coordinates (without replacement) with size equal to the number of individuals 
    Arguments: number_of_players - the number of coordinates in the returned list
                size_of_lattice - a number that determines the size of the lattice (it is a square)

    returns: a list of coordinates (i.e. tuples) 
    '''
    choice = np.random.choice(size_of_lattice**2, number_of_players, replace=False)
    locations = []
    for i in range(size_of_lattice):
        for j in range(size_of_lattice): 
            locations.append((i,j))

    choice_locations = []
    for i in choice: 
        choice_locations.append(locations[i])

    return choice_locations

def initiate_simulation(SETTINGS):
    '''Takes in the SETTINGS dicitionary and outputs a list of individuals and a lattice with the specified charateristics
        
        inputs: 
            SETTINGS - the dicitionary with the global settings
        
        outputs: 
            a tuple with the a dicitionary a indiviuals and a lattice
    '''
    number_of_individuals = SETTINGS['Number of individuals']
    size_of_lattice = SETTINGS['Size of lattice']
    starting_sugar_level = SETTINGS['Starting sugar level']
    proportion_of_hawks = SETTINGS['Proportion of agressors']
    sight = SETTINGS['Individual sight']
    metabolism = SETTINGS['Metabolism']
    bias_in_favour_of_attacker = SETTINGS['Bias in favour of attacker'] 
    
    if 0 > proportion_of_hawks or 1 < proportion_of_hawks: 
        print('proportion_of_hawks should be a proprtion, i.e. a number between 0 and 1')

    individuals = dict()
    number_of_hawks = round(proportion_of_hawks*number_of_individuals)
    number_of_doves = number_of_individuals - number_of_hawks
    initial_locations = random_initial_locations(number_of_individuals, size_of_lattice) # Chooses random initial location

    # Generate strengths
    parameter_1 = SETTINGS['Strength distribution']['Parameter 1']
    parameter_2 = SETTINGS['Strength distribution']['Parameter 2']
    if SETTINGS['Strength distribution']['Family'] == 'Gamma':  
        strengths = np.random.gamma(parameter_1, parameter_2, size=number_of_individuals)
    elif SETTINGS['Strength distribution']['Family'] == 'Uniform': 
        strengths = np.random.uniform(parameter_1, parameter_2, size=number_of_individuals)
    elif SETTINGS['Strength distribution']['Family'] == 'Exponential': 
        strengths = numpy.random.exponential(parameter_1, size=number_of_individuals)

    for n in range(number_of_individuals):
        k = np.random.randint(1, 3000 - n) # For names
        name = names[k]
        names.remove(name)
        location = initial_locations[n]
        strength = strengths[n]
        if n < number_of_hawks:
            agressor_type = "hawk"
        else: 
            agressor_type = "dove"
        individuals[name] = Individual(starting_sugar_level, sight, metabolism, location, name, agressor_type, strength, proportion_of_hawks, bias_in_favour_of_attacker)

    lattice = Lattice(SETTINGS)
    return individuals, lattice

def play_round_combat(individuals, dead_individuals, combat_log, round_number, lattice, SETTINGS): 
    '''Plays one round of the simlutaion with combats and coalition
    Arguments:  individuals - a dictionary of individuals
                lattice - the lattice
                individuals - a dicitionary of all individuals
                dead_individuals - a dicitionary of all individuals that have died (good to keep track of for debugging purposes)
                combat_log - a dictionary with information on all the combats, good for debugginf purposes
                round_number - the round number (needed to store data properly)
                lattice - the lattice of the simulation
                SETTINGS - a dicitonary with all the global settings
    '''

    # Start with shuffling the order in which the individuals get to make their moves. So as to not advantage any particular individaul. 
    l = list(individuals.items())
    random.shuffle(l)
    individuals = dict(l)

    cost_of_combat = SETTINGS['Cost of combat']
    bias_in_favour_of_attacker = SETTINGS['Bias in favour of attacker']
    majority = (SETTINGS['Voting Mechanism'])

    for k,v in individuals.items():
        nbhd = v.find_nbhd(individuals)
        v.decide_on_coalition_membership(individuals, nbhd) 
        if v.type == "hawk":
            v.move_hawk(lattice, individuals, nbhd, cost_of_combat, bias_in_favour_of_attacker, combat_log, round_number, SETTINGS)
        else:
            v.move(individuals, lattice)
        v.harvest(lattice)
        v.eat()

    lattice.regrow_sugar() 

### PLOTS AND DATA ### 
'''Below follows some funciton with the purpose to store data and plot the outputs
'''

# plotting the location of all the individauls
def plot_locations(individuals, round_number, combat_log, show_combat_log=True):
    '''Plot the location of the living individuals using a scatter plot
    Arguments: Individuals - dictionary of individuals
    '''  

    list_of_coordinates = []
    for k,v in individuals.items():
        if v.alive == True: 
            loc = v.location
            list_of_coordinates.append(loc)
    if len(list_of_coordinates) > 0:
        x, y = zip(*list_of_coordinates)
        plt.scatter(x, y, marker='x')
        for k,v in individuals.items(): 
            if v.alive == True:
                x,y = v.location
                strength = v.find_strength_of_coalition(v.coalition, individuals, nbhd)
                sugar = v.find_sugar_of_coalition(v.coalition, individuals, nbhd)
                prob_attack = v.assess_probability_of_attack_individual(strength, sugar, individuals, nbhd)
                plt.text(x,y, v.name + "\n" + v.type + "\n" + 'strength: ' + str(strength) + "\n" + 'sugar: ' + str(sugar) + "\n" + str(prob_attack) + "\n" + str(v.coalition.number))
        plt.title('Locations, Type and Sugar\nRound: ' + str(round_number))
        if show_combat_log:
            combat_number = 1
            log_entry = ""
            print(round_number)
            for i in combat_log['round ' + str(round_number)]:
                ith_combat = "Combat " + str(combat_number) + ": " + i["Attacker"] + ' attacked ' + i["Attacked"] + ', ' + i["Winner"] + ' won.' 
                log_entry += ith_combat
                combat_number += 1
            plt.xlabel(log_entry)
        if not export_plots: 
            plt.show()
        plt.clf()
    else: 
        print('Could not plot the location of the alive individuals since all individuals are dead.')

def plot_sugar(lattice): 
    ''' Takes in the lattice and plots the sugar levels at all points of the lattice using an inverse heatmap
    Arguments: lattice - the lattice in question
    '''
    a = lattice.sugar
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(a, linewidth=0.5,cmap = cmap)

    if not export_plots: 
        plt.show()
    plt.clf()

def plot_coalitions_ii(individuals, size): 
    n = size
    a = np.zeros((n, n))

    for k,v, in individuals.items(): 
        loc = v.location
        coalition_number = v.coalition.number
        a[loc] = coalition_number

    cmap = sns.cm.rocket_r
    c = np.random.rand(3,)
    ax = sns.heatmap(a, linewidth=0.5,cmap=c)
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'coalition_plot.pgf')
        plt.clf()
    else: 
        plt.show()

def find_spatial_dominance_of_coalition(coalition, individuals):
    all_individuals_in_area_w_duplicates = []
    for i in coalition.members:
        for j in i.find_nbhd(individuals): 
            all_individuals_in_area_w_duplicates.append(j)
    all_individuals_in_area = list(set(all_individuals_in_area_w_duplicates))
    spatial_dominance = len(coalition.members)/len(all_individuals_in_area)
    return spatial_dominance

def find_greatest_coalition(coalitions): 
    greatest_coalition = coalitions[0]
    greatest_number_of_members = 1
    for c in coalitions:
        x = len(c.members)
        if x > greatest_number_of_members:
            greatest_coalition = c
            greatest_number_of_members = x
    return greatest_coalition

def who_is_alive_and_dead(individuals):
    alive_individuals = []
    dead_individuals = [] 

    for k,v in individuals.items(): 
        if v.alive == True:
            alive_individuals.append(v)
        else: 
            dead_individuals.append(v)
    return alive_individuals, dead_individuals

def plot_time_series_average_resource_level(mean_sugar_doves, mean_sugar_hawks):
    x_vaules = list(range(len(mean_sugar_hawks))) 
    plt.plot(x_vaules, mean_sugar_doves, linewidth=1.0, label='Cooperators')
    plt.plot(x_vaules, mean_sugar_hawks, linewidth=1.0, label='Agressors')
    plt.xlabel('Time')
    plt.ylabel('Average Resource Level')
    plt.legend()
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'average_resource_level.pgf')
        plt.clf()
    else:
        plt.show()

def plot_time_series_average_strength_coalitions(average_strength_coalitions):
    x_vaules = list(range(len(average_strength_coalitions))) 
    plt.plot(x_vaules, average_strength_coalitions, linewidth=1.0, label='Average Power for Coalition')
    plt.xlabel('Time')
    plt.ylabel('Average Power Coalitions')
    plt.legend()
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'average_power.pgf')
        plt.clf()
    else:
        plt.show()

def plot_time_series_conflicts(number_of_conflicts):
    x_vaules = list(range(len(number_of_conflicts))) 
    plt.plot(x_vaules, number_of_conflicts, linewidth=1.0, label='Number of conflicts')
    plt.xlabel('Time')
    plt.ylabel('Number of conflicts')
    plt.legend()
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'number_of_conflict.pgf')
        plt.clf()
    else:
        plt.show()

def plot_time_series_dominance(spatial_dominance_of_greatest_coalition):
    x_vaules = list(range(len(spatial_dominance_of_greatest_coalition))) 
    plt.plot(x_vaules, spatial_dominance_of_greatest_coalition, linewidth=1.0, label='Spatial dominance of greatest association')
    plt.xlabel('Time')
    plt.ylabel('Percentage')
    plt.legend()
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'spatial_dominance.pgf')
        plt.clf()
    else:
        plt.show()

def plot_dominance_and_average_strength(spatial_dominance_of_greatest_coalition, average_strength_coalitions):
    x_vaules = list(range(len(average_strength_coalitions)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.375), tight_layout=True)

    axs = {ax1: [spatial_dominance_of_greatest_coalition, 'Spatial Dominance of Greatest Association'], ax2: [average_strength_coalitions, "Average Power Association"]}

    for ax, data in axs.items(): 
        ax.plot(x_vaules, data[0], linewidth=1.0, label=data[1])
        ax.set_xlabel('Time')
        ax.set_ylabel(data[1])
        #ax.legend()
    
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'average_power_and_spatial_dominance.pgf')
        plt.clf()
    else:
        plt.show()

def plot_conflicts_and_resource_level(number_of_conflicts, mean_sugar_doves, mean_sugar_hawks):
    x_vaules = list(range(len(number_of_conflicts)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.375), tight_layout=True)
  
    ax1.plot(x_vaules, number_of_conflicts, linewidth=1.0, label='Number of Conflicts#')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Conflicts')
    ax2.plot(x_vaules, mean_sugar_doves, linewidth=1.0, label="Cooperators")
    ax2.plot(x_vaules, mean_sugar_hawks, linewidth=1.0, label="Agressors")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Average Resource Level')
    ax2.legend()
    
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'conflicts_and_resource_level.pgf')
        plt.clf()
    else:
        plt.show()


def plot_coalitions_iii(data_1, data_2):
    N,N = data_1[0].shape

    # create discrete colormap
    cmap = colors.ListedColormap(['blue', 'grey', 'yellow', 'red', 'white', 'orange', 'purple'])
    bounds = [0,0,1,2,3,4,5,6,7]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.375), tight_layout=True)

    axs = {ax1: data_1, ax2: data_2}

    for ax, data in axs.items(): 
    # draw gridlines
        ax.imshow(data[0], cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='white', linewidth=0.25)
        ax.set_xticks(np.arange(-.5, N, 1));
        ax.set_yticks(np.arange(-.5, N, 1));
        plt.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        ax.set_title(data[1]) 
        ax.tick_params(tick1On=False)
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'double_coalitions.pgf')
        plt.clf()
    else:
        plt.show()
    
def plot_coalitions_iiii(data_1, data_2, data_3, data_4):
    '''
    data_* - An ordered pair with a np.array as its first element and the relevant round number as its second
    '''

    N,N = data_1[0].shape

    # create discrete colormap
    cmap = colors.ListedColormap(['blue', 'grey', 'yellow', 'red', 'white', 'orange', 'purple'])
    bounds = [0,0,1,2,3,4,5,6,7]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(6.75, 3.375), tight_layout=True)

    axs = {ax1: data_1, ax2: data_2, ax3: data_3, ax4: data_4}

    for ax, data in axs.items(): 
    # draw gridlines
        ax.imshow(data[0], cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='white', linewidth=0.25)
        ax.set_xticks(np.arange(-.5, N, 1));
        ax.set_yticks(np.arange(-.5, N, 1));
        plt.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        ax.tick_params(tick1On=False)
        ax.set_title(data[1]) 
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'quadruple_coalitions.pgf')
        plt.clf()
    else:
        plt.show()

def plot_coalitions_v(data):
    '''
    data_* - An ordered pair with a np.array as its first element and the relevant round number as its second
    '''

    N,N = data[0].shape

    # create discrete colormap
    cmap = colors.ListedColormap(['blue', 'grey', 'yellow', 'red', 'white', 'orange', 'purple'])
    bounds = [0,0,1,2,3,4,5,6,7]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(6.75, 3.375), tight_layout=True)

    # draw gridlines
    ax.imshow(data[0], cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='white', linewidth=0.25)
    ax.set_xticks(np.arange(-.5, N, 1));
    ax.set_yticks(np.arange(-.5, N, 1));
    plt.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    ax.set_title(data[1]) 
    if export_plots: 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        plt.savefig(date_time + 'single_coalitions.pgf')
        plt.clf()
    else:
        plt.show()

def save_prop_hawks(individuals):
    alive_individuals, dead_individuals = who_is_alive_and_dead(individuals)
    number_of_alive_individuals = len(alive_individuals)
    alive_hawks = list(filter(lambda type: type == 'hawk', [v.type for v in alive_individuals]))
    number_of_alive_hawks = len(alive_hawks)
    return number_of_alive_hawks/number_of_alive_individuals

def save_mean_sugar_hawks(individuals):
    alive_individuals, dead_individuals = who_is_alive_and_dead(individuals)
    alive_hawks = list(filter(lambda v: v.type == 'hawk', alive_individuals))
    mean_sugar_hawks = sum(map(lambda k: k.sugar, alive_hawks))/len(alive_hawks)
    return mean_sugar_hawks

def save_mean_sugar_doves(individuals):
    alive_individuals, dead_individuals = who_is_alive_and_dead(individuals)
    alive_doves = list(filter(lambda v: v.type == 'dove', alive_individuals))
    mean_sugar_doves = sum(map(lambda k: k.sugar, alive_doves))/len(alive_doves)
    plt.savefig('histogram.pgf')
    return mean_sugar_doves

def save_coalition_data(individuals, size, round_number): 
    coalitions = []
    for k,v in individuals.items():
        coalitions.append(v.coalition)
    coalitions = list(set(coalitions))

    data = np.zeros((size, size))
    n = len(individuals)
    counter = 2
    large_coalitions_dict = {}
    for c in coalitions: 
        if len(c.members)/n >= 0.1:
            large_coalitions_dict[c] = counter
            counter +=1

    for k,v in individuals.items(): 
        try: 
            x = large_coalitions_dict[v.coalition]
            data[v.location] = x 
        except: 
            data[v.location] = 1

    label = 'Round ' + str(round_number)
    return (data, label)

def save_number_of_conlficts(combat_log, round_number):
    number_of_conflicts = len(combat_log['round ' + str(round_number)])
    return number_of_conflicts

def save_average_strength_coalitions(individuals):
    coalitions = []
    for k,v in individuals.items():
        coalitions.append(v.coalition)
    coalitions = list(set(coalitions))

    total_strength = 0
    for c in coalitions:
        c_strength = sum(i.strength for i in c.members)
        total_strength += c_strength
    average = total_strength/len(coalitions)
    return average

def print_all_coalitions(individuals): 
    coalitions = []
    for k,v in individuals.items():
        coalitions.append(v.coalition)

    coalitions = list(set(coalitions))
    for c in coalitions: 
        print(c.members)

    return coalitions

# CREATING SOME LISTS TO SAVE THE DATA
prop_hawks = []
spatial_dominance_of_greatest_coalition = []
mean_sugar_hawks = []
mean_sugar_doves = []
mean_strength = []
number_of_conflicts = []
dead_individuals = {}
round_number = 0
combat_log = {}

### RUN THE SIMULATION
individuals, lattice = initiate_simulation(SETTINGS)
no_of_rounds = SETTINGS['Number of rounds']

# For plotting the coalitions spatial structure 
if no_of_rounds % 40 == 0: 
    middle_middle_round = int(no_of_rounds/40)
else: 
    middle_middle_round = int(math.ceil(no_of_rounds/10)) 
if no_of_rounds % 2 == 0: 
    middle_round = int(no_of_rounds/10)
else: 
    middle_round = int(math.ceil(no_of_rounds/10)) 

for i in tqdm(range(no_of_rounds)): 
    round_number += 1
    # For plotting coalitions halfway through the simulation 
    
    combat_log['round ' + str(round_number)] = []
    play_round_combat(individuals, dead_individuals, combat_log, round_number, lattice, SETTINGS) 
    morgue = []
    for k,v in individuals.items(): 
        if not v.alive: 
            morgue.append(v.name)
    for i in morgue:
        dead_individuals[i] = individuals.pop(i)
    
    if round_number ==  1:
        coalitions_part_1 = save_coalition_data(individuals, lattice.size, round_number)

    if round_number ==  middle_middle_round:
        coalitions_part_2 = save_coalition_data(individuals, lattice.size, round_number)

    if round_number ==  middle_round:
        coalitions_part_3 = save_coalition_data(individuals, lattice.size, round_number)

    if round_number ==  no_of_rounds:
        coalitions_part_4 = save_coalition_data(individuals, lattice.size, round_number)

    coalitions = []
    for k,v in individuals.items():
        coalitions.append(v.coalition)
    coalitions = list(set(coalitions))

    prop_hawks.append(save_prop_hawks(individuals))
    mean_sugar_hawks.append(save_mean_sugar_hawks(individuals)) 
    mean_sugar_doves.append(save_mean_sugar_doves(individuals))
    mean_strength.append(save_average_strength_coalitions(individuals))
    number_of_conflicts.append(save_number_of_conlficts(combat_log, round_number))
    gc = find_greatest_coalition(coalitions)
    spd = find_spatial_dominance_of_coalition(gc,individuals)
    spatial_dominance_of_greatest_coalition.append(spd)
    


### PLOTTING THE RESULTS
plot_coalitions_v(coalitions_part_4)
plot_conflicts_and_resource_level(number_of_conflicts, mean_sugar_doves, mean_sugar_hawks)
plot_dominance_and_average_strength(spatial_dominance_of_greatest_coalition, mean_strength)
plot_time_series_dominance(spatial_dominance_of_greatest_coalition)
plot_coalitions_iii(coalitions_part_3,  coalitions_part_4)
plot_coalitions_iiii(coalitions_part_1,  coalitions_part_2, coalitions_part_3,  coalitions_part_4) 

