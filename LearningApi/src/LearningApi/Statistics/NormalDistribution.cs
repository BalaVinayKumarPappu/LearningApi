﻿
namespace LearningFoundation.Statistics
{
    using System;
    //using Accord.Math;
    //using Accord.Statistics.Distributions;
    //using Accord.Statistics.Distributions.Fitting;
    //using Accord.Statistics.Distributions.Multivariate;
    //using AForge;
    
    public class NormalDistribution 
    {

        // Distribution parameters
        private double mean = 0;   // mean μ
        private double stdDev = 1; // standard deviation σ

        // Distribution measures
        private double? entropy;

        // Derived measures
        private double variance = 1; // σ²
        private double lnconstant;   // log(1/sqrt(2*pi*variance))

        private bool immutable;

        // 97.5 percentile of standard normal distribution
        private const double p95 = 1.95996398454005423552;

     
        /// <summary>
        ///   Generates a single random observation from the 
        ///   Normal distribution with the given parameters.
        /// </summary>
        /// 
        /// <param name="mean">The mean value μ (mu).</param>
        /// <param name="stdDev">The standard deviation σ (sigma).</param>
        ///
        /// <returns>An double value sampled from the specified Normal distribution.</returns>
        /// 
        public static double Random(double mean, double stdDev)
        {
            return Random() * stdDev + mean;
        }

        /// <summary>
        ///   Generates a random vector of observations from the 
        ///   Normal distribution with the given parameters.
        /// </summary>
        /// 
        /// <param name="mean">The mean value μ (mu).</param>
        /// <param name="stdDev">The standard deviation σ (sigma).</param>
        /// <param name="samples">The number of samples to generate.</param>
        ///
        /// <returns>An array of double values sampled from the specified Normal distribution.</returns>
        /// 
        public static double[] Random(double mean, double stdDev, int samples)
        {
            return Random(mean, stdDev, samples, new double[samples]);
        }

        /// <summary>
        ///   Generates a random vector of observations from the 
        ///   Normal distribution with the given parameters.
        /// </summary>
        /// 
        /// <param name="mean">The mean value μ (mu).</param>
        /// <param name="stdDev">The standard deviation σ (sigma).</param>
        /// <param name="samples">The number of samples to generate.</param>
        /// <param name="result">The location where to store the samples.</param>
        ///
        /// <returns>An array of double values sampled from the specified Normal distribution.</returns>
        /// 
        public static double[] Random(double mean, double stdDev, int samples, double[] result)
        {
            Random(samples, result);
            for (int i = 0; i < samples; i++)
                result[i] = result[i] * stdDev + mean;
            return result;
        }

        [ThreadStatic]
        private static bool useSecond = false;

        [ThreadStatic]
        private static double secondValue = 0;

        /// <summary>
        ///   Generates a random vector of observations from the standard
        ///   Normal distribution (zero mean and unit standard deviation).
        /// </summary>
        /// 
        /// <param name="samples">The number of samples to generate.</param>
        /// <param name="result">The location where to store the samples.</param>
        ///
        /// <returns>An array of double values sampled from the specified Normal distribution.</returns>
        /// 
        public static double[] Random(int samples, double[] result)
        {
            
            var rand = LearningFoundation.Math.Random.Generator.Random;

            bool useSecond = NormalDistribution.useSecond;
            double secondValue = NormalDistribution.secondValue;

            for (int i = 0; i < samples; i++)
            {
                // check if we can use second value
                if (useSecond)
                {
                    // return the second number
                    useSecond = false;
                    result[i] = secondValue;
                    continue;
                }

                // Polar form of the Box-Muller transformation
                // http://www.design.caltech.edu/erik/Misc/Gaussian.html

                double x1, x2, w, firstValue;

                // generate new numbers
                do
                {
                    x1 = rand.NextDouble() * 2.0 - 1.0;
                    x2 = rand.NextDouble() * 2.0 - 1.0;
                    w = x1 * x1 + x2 * x2;
                }
                while (w >= 1.0);

                w = Math.Sqrt((-2.0 * Math.Log(w)) / w);

                // get two standard random numbers
                firstValue = x1 * w;
                secondValue = x2 * w;

                useSecond = true;

                // return the first number
                result[i] = firstValue;
            }

            NormalDistribution.useSecond = useSecond;
            NormalDistribution.secondValue = secondValue;

            return result;
        }

        /// <summary>
        ///   Generates a random value from a standard Normal 
        ///   distribution (zero mean and unit standard deviation).
        /// </summary>
        /// 
        public static double Random()
        {
            var rand = LearningFoundation.Math.Random.Generator.Random;

            // check if we can use second value
            if (useSecond)
            {
                // return the second number
                useSecond = false;
                return secondValue;
            }

            double x1, x2, w, firstValue;

            // generate new numbers
            do
            {
                x1 = rand.NextDouble() * 2.0 - 1.0;
                x2 = rand.NextDouble() * 2.0 - 1.0;
                w = x1 * x1 + x2 * x2;
            }
            while (w >= 1.0);

            w = Math.Sqrt((-2.0 * Math.Log(w)) / w);

            // get two standard random numbers
            firstValue = x1 * w;
            secondValue = x2 * w;

            useSecond = true;

            // return the first number
            return firstValue;
        }
    }
}