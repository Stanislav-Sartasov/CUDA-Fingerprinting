using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.FeatureExtraction.Minutiae
{
    class MinutiaeMatching
    {
        private static float SumRow(float[,] s, int row)
        {
            float sum = 0;

            for (int j = 0; j < s.GetLength(1); ++j)
            {
                sum += s[row, j];
            }

            return sum;
        }

        private static float SumColumn(float[,] s, int column)
        {
            float sum = 0;

            for (int i = 0; i < s.GetLength(0); ++i)
            {
                sum += s[i, column];
            }

            return sum;
        }

        private static void Normalize(float[,] s, List<Minutia> mins1, List<Minutia> mins2)
        {
            float[] rowSum = new float[s.GetLength(0)];
            float[] columnSum = new float[s.GetLength(1)];
            float fengAngle = 1.74532925F;

            for (int i = 0; i < s.GetLength(0); ++i)
            {
                rowSum[i] = SumRow(s, i);
            }

            for (int j = 0; j < s.GetLength(1); ++j)
            {
                columnSum[j] = SumColumn(s, j);
            }

            for (int i = 0; i < s.GetLength(0); ++i)
            {
                for (int j = 0; j < s.GetLength(1); ++j)
                {
                    if (Math.Abs(mins1[i].Angle - mins2[j].Angle) < fengAngle)
                    {
                        s[i, j] = s[i, j] * (s.GetLength(0) + s.GetLength(1) - 1) / (rowSum[i] + columnSum[j] - s[i, j]);
                    }
                    else
                    {
                        s[i, j] = 0;
                    }
                }
            }
        }

        private static List<Tuple<float, int, int>> ArrayToTupleList(float[,] s)
        {
            List<Tuple<float, int, int>> l = new List<Tuple<float, int, int>>(s.GetLength(0) * s.GetLength(1));

            for (int i = 0; i < s.GetLength(0); ++i)
            {
                for (int j = 0; j < s.GetLength(1); ++j)
                {
                    l.Add(new Tuple<float, int, int>(s[i, j], i, j));
                }
            }

            l.Sort((x, y) => y.Item1.CompareTo(x.Item1));

            return l;
        }

        private static bool isMatchable(/*Minutia m1, Minutia m2, Minutia kernel1, Minutia kernel2*/)
        {

            return true;
        }

        public static List<List<Tuple<int, int>>> MatchMinutiae(float[,] s, List<Minutia> mins1, List<Minutia> mins2)
        {
            int top = 10;
            List<List<Tuple<int, int>>> res = new List<List<Tuple<int, int>>>(top);
            bool[] flag1 = new bool[s.GetLength(0)];
            bool[] flag2 = new bool[s.GetLength(1)];
            int i0, j0, i, j;
            List<Tuple<int, int>> temp = new List<Tuple<int, int>>();

            Normalize(s, mins1, mins2);
            List<Tuple<float, int, int>> list = ArrayToTupleList(s);

            for (int k = 0; k < top; ++k)
            {
                i0 = list[k].Item2;
                j0 = list[k].Item3;

                Array.Clear(flag1, 0, flag1.GetLength(0));
                Array.Clear(flag2, 0, flag2.GetLength(0));
                temp = new List<Tuple<int, int>>();

                flag1[i0] = true;
                flag2[j0] = true;
                
                for (int m = 0; m < list.Count; m++)
                {
                    i = list[m].Item2;
                    j = list[m].Item3; 

                   if (!flag1[i] && !flag2[j] && isMatchable())
                   {
                       temp.Add(new Tuple<int, int>(i, j));

                       flag1[i] = true;
                       flag2[j] = true;
                   }
                }

                res.Add(temp);
            }

            return res;
        }
    }
}

