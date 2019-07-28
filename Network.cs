using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



namespace mlp_net_ga_cs
{
    class Network
    {
        public int inputs, hiddens, outputs;
        public Matrix inputWeights, hiddenWeights;
        

        Network(ref Random rnd,in int input, in int hidden, in int output)
        {

            this.inputs = input;
            this.hiddens = hidden;
            this.outputs = output;

            this.inputWeights = new Matrix();
            this.hiddenWeights = new Matrix();

            inputWeights.RandomMatrix(ref rnd, in hidden, in input);
            hiddenWeights.RandomMatrix(ref rnd, in output, in hidden);

        }

        public Matrix Predict(in double[] inputData)
        {
            Matrix inputs = new Matrix();
            int l = inputData.Length;
            inputs.EmptyMatrix(l, 1);
            for (int i = 0; i < l; i++)
            {
                inputs.matrix[i, 0] = inputData[i];
            }

            Matrix hiddenInputs = inputWeights.DotProduct(in inputs);
            Matrix hiddenOutputs = hiddenInputs.Sigmoid();
            Matrix finalInputs = hiddenWeights.DotProduct(in hiddenOutputs);
            Matrix finalOutputs = finalInputs.Sigmoid();

            return finalOutputs;

        }

        static void Main(string[] args)
        {
            Random rnd = new Random();
            Console.Write("Creating Network");
            Console.Write(Environment.NewLine);

            Network net = new Network(ref rnd,2,4,1);

            Console.Write("Initialising Trainer");
            Trainer trainer = new Trainer(in net, 500, 1, 200, 3, 5);

            Console.Write("Training Network");
            trainer.Train(ref rnd,ref net, 10000, true);

            double[] inputone = { 0.0, 0.0 };
            Matrix.printMatrix(net.Predict(in inputone));

            double[] inputtwo = { 1.0, 0.0 };
            Matrix.printMatrix(net.Predict(in inputtwo));

            double[] inputthree = { 0.0, 1.0 };
            Matrix.printMatrix(net.Predict(in inputthree));

            double[] inputfour = { 1.0, 1.0 };
            Matrix.printMatrix(net.Predict(in inputfour));

            Console.ReadKey();
        }
    }
}
