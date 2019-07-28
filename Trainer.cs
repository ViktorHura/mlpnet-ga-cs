using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Medallion;

namespace mlp_net_ga_cs
{
    class Trainer
    {
        int popsize, elite, cutoff, tournamentsize, inputs, hiddens, outputs;
        double mutationrate;
        Organism[] population;

        public Trainer(in Network net,in int pops,in int el,in int cutf,in int tournsize,in double mutrate)
        {
            population = new Organism[pops];
            popsize = pops;
            elite = el;
            cutoff = cutf;
            tournamentsize = tournsize;
            mutationrate = mutrate;

            inputs = net.inputs;
            hiddens = net.hiddens;
            outputs = net.outputs;
        }

        public void Train(ref Random rnd,ref Network net,in int maxgen,in Boolean reset)
        {
            if (reset == true)
            {
                this.CreateInitialPopulation(ref rnd);
            }

            for (int generation = 0; generation < maxgen; generation++)
            {



                for (int i = 0; i < popsize; i++)
                {
                    population[i].CalcFitness(ref net);
                }

                
                this.SortPopulation();
                Console.Write("Gen: " + generation + " Bfit: " + population[0].fitness);
                Console.Write(Environment.NewLine);




                this.NaturalSelection(ref rnd);






            }

            this.SortPopulation();
            net.inputWeights = population[0].inputWeights;
            net.hiddenWeights = population[0].hiddenWeights;

        }





        public void CreateInitialPopulation(ref Random rnd)
        {
            for (int i = 0; i < popsize; i++)
            {
                Organism o = new Organism(inputs, hiddens, outputs, ref rnd);
                population[i] = o;
            }
        }

        public void NaturalSelection(ref Random rnd)
        {
            Organism[] nextPopulation = new Organism[popsize];


            for (int i = 0; i < popsize; i++)
            {
                if (i < elite)
                {
                    nextPopulation[i] = population[i];
                    continue;
                }
                else
                {
                    Organism a = this.TournamentSelection(ref rnd);
                    Organism b = this.TournamentSelection(ref rnd);

                    Organism child = a.Crossover(ref rnd,in b);
                    child.Mutate(ref rnd, in mutationrate);

                    child.fitness = 0.0;

                    nextPopulation[i] = child;

                }

            }

            population = nextPopulation;



        }

        public void SortPopulation()
        {
            Array.Sort(population, delegate (Organism o1, Organism o2) {
                return o2.fitness.CompareTo(o1.fitness); // element 0 highest fitness
            });
        }

        public Organism TournamentSelection(ref Random rnd)
        {
            
            Organism[] tournament = new Organism[tournamentsize];

            for (int n = 0; n < tournamentsize; n++)
            {
                int i = (int)(rnd.NextDouble(1) * (popsize - cutoff));
                tournament[n] = population[i];

            }

            int bestint = 0;
            double bestfit = tournament[0].fitness;

            for (int n = 0; n < tournamentsize; n++)
            {
                if (tournament[n].fitness > bestfit)
                {
                    bestint = n;
                    bestfit = tournament[n].fitness;
                }
            }

            return tournament[bestint];

        }
    }
}

    


