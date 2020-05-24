class GeneticSelector():
    def __init__(self, estimator, generation, size, best, random, 
                 children, mutation_r):
        # Estimator 
        self.estimator = estimator
        # Number of generations
        self.generation = generation
        # Number of chromosomes in population
        self.size = size
        # Number of best chromosomes to select
        self.best = best
        # Number of random chromosomes to select
        self.random = random
        # Number of children created during crossover
        self.children = children
        # Probablity of chromosome mutation
        self.mutation_r = mutation_r
        
        if int((self.best + self.random) / 2) * self.children != self.size:
            raise ValueError("The population size is not stable.")  
            
    def initilize(self):
        population = []
        for i in range(self.size):
            chromosome = np.ones(self.n_features, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < 0.3
            chromosome[mask] = False
            population.append(chromosome)
        return population

    def fitness(self, population):
        X, y = self.dataset
        scores = []
        for chromosome in population:
            score = -1.0 * np.mean(cross_val_score(self.estimator, X[:,chromosome], y, 
                                                       cv=5, 
                                                       scoring="neg_mean_squared_error"))
            scores.append(score)
        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)
        return list(scores[inds]), list(population[inds,:])

    def select(self, population_sorted):
        population_next = []
        for i in range(self.best):
            population_next.append(population_sorted[i])
        for i in range(self.random):
            population_next.append(random.choice(population_sorted))
        random.shuffle(population_next)
        return population_next

    def crossover(self, population):
        population_next = []
        for i in range(int(len(population)/2)):
            for j in range(self.children):
                chromosome1, chromosome2 = population[i], population[len(population)-1-i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.5
                child[mask] = chromosome2[mask]
                population_next.append(child)
        return population_next
	
    def mutate(self, population):
        population_next = []
        for i in range(len(population)):
            chromosome = population[i]
            if random.random() < self.mutation_r:
                mask = np.random.rand(len(chromosome)) < 0.05
                chromosome[mask] = False
            population_next.append(chromosome)
        return population_next

    def generate(self, population):
        # Selection, crossover and mutation
        scores_sorted, population_sorted = self.fitness(population)
        population = self.select(population_sorted)
        population = self.crossover(population)
        population = self.mutate(population)
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))
        
        return population

    def fit(self, X, y):
 
        self.chromosomes_best = []
        self.scores_best, self.scores_avg  = [], []
        
        self.dataset = X, y
        self.n_features = X.shape[1]
        
        population = self.initilize()
        for i in range(self.generation):
            population = self.generate(population)
            
        return self 
    
    @property
    def support_(self):
        return self.chromosomes_best[-1]

    def plot_scores(self):
        plt.plot(self.scores_best, label='Best')
        plt.plot(self.scores_avg, label='Average')
        plt.legend()
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        plt.show()
#import necessary liraries to work with
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv('Admission_Predict.csv')
data.columns
SEED = 2000
random.seed(SEED)
np.random.seed(SEED)


#data sampling


train, test = train_test_split(data, test_size=0.2)

X=data[['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']].values
y=data['Chance of Admit '].values

estimator_f = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#df1 = df.head(25)
#df.head(25)
#df1.plot(kind='bar',figsize=(10,8))
#plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)*1000)  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

selector = GeneticSelector(estimator=LinearRegression(), 
                      generation=4, size=200, best=40, random=40, 
                      children=5, mutation_r=0.05)
selector.fit(X_train, y_train)
selector.plot_scores()
score = -1.0 * cross_val_score(est, X_test[:,selector.support_], y_pred, cv=5, scoring="neg_mean_squared_error")
print('Mean Squared Error before doing feature selection:', metrics.mean_squared_error(y_test, y_pred)*1000)
print("Mean Square Error after feature selection: {:.2f}".format(np.mean(score)*10000))
