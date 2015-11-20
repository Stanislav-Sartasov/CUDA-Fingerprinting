using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NonParrallelVersion
{
    class AlgorythmOPATA
    {
        private static Move[] offsets =
        {
            new Move(-1,0), 
            new Move(1,0),
            new Move(0,-1),
            new Move(0,1)
        };

        struct Move
        {
            internal int XOffset;
            internal int YOffset;

            public Move(int x, int y)
            {
                XOffset = x;
                YOffset = y;
            }
        }

        private string IsComparedAN(int[][] A, int p8, int p9)
        {
            int y = 0;

            int[][] a = new int[][] { new int[]{ 1, 1, y }, 
                                      new int[] { 1, A[1][1], 0}, 
                                      new int[] { 1, 1, y } };


            if ((A[0][2] == 0) || (A[2][2] == 0))
            {
                a[0][2] = A[0][2];
                a[2][2] = A[2][2];
                if ((A[0].SequenceEqual(a[0])) && (A[1].SequenceEqual(a[1])) && (A[2].SequenceEqual(a[2])))
                    return "a";
            }

            int[][] b = new int[][] { new int[]{ 1, 1, 1 }, 
                                      new int[] { 1, A[1][1], 1 }, 
                                      new int[] { y, 0, y } };
            if ((A[2][0] == 0) || (A[2][2] == 0))
            {
                b[2][0] = A[2][0];
                b[2][2] = A[2][2];
                if ((A[0].SequenceEqual(b[0])) && (A[1].SequenceEqual(b[1])) && (A[2].SequenceEqual(b[2])))
                    return "b";
            }

            int[][] c = new int[][] { new int[]{ y, 1, 1}, 
                                      new int[] { 0, A[1][1], 1}, 
                                      new int[]{ y, 1, 1} };
            if (((A[0][0] == 0) || (A[2][0] == 0)) && (p8 == 1))
            {
                c[0][0] = A[0][0];
                c[2][0] = A[2][0];

                if ((A[0].SequenceEqual(c[0])) && (A[1].SequenceEqual(c[1])) && (A[2].SequenceEqual(c[2])))
                    return "c";
            }

            int[][] d = new int[][] {new int[] { y, 0, y }, 
                                    new int[]{ 1, A[1][1], 1 }, 
                                    new int[]{ 1, 1, 1 }};
            if (((A[0][0] == 0) || (A[0][2] == 0)) && (p9 == 1))
            {
                d[0][0] = A[0][0];
                d[0][2] = A[0][2];

                if ((A[0].SequenceEqual(d[0])) && (A[1].SequenceEqual(d[1])) && (A[2].SequenceEqual(d[2])))
                    return "d";
            }

            int[][] e = new int[][] { new int[]{ A[0][0], 0, 0 }, 
                                      new int[] { 1, A[1][1], 0 }, 
                                      new int[] { A[0][2], 1, A[2][2] } };
            if ((A[0].SequenceEqual(e[0])) && (A[1].SequenceEqual(e[1])) && (A[2].SequenceEqual(e[2])))
                return "e";

            int[][] f = new int[][] { new int[]{ A[0][0], 1, 1 }, 
                                   new int[] { 0, A[1][1], 1 }, 
                                   new int[] { 0, 0, A[2][2] } };
            if ((A[0].SequenceEqual(f[0])) && (A[1].SequenceEqual(f[1])) && (A[2].SequenceEqual(f[2])))
                return "f";

            int[][] g = new int[][] { new int[]{ 0, 1, 0 }, 
                                      new int[] { 0, A[1][1], 1 }, 
                                      new int[]{ 0, 0, 0 } };
            if ((A[0].SequenceEqual(g[0])) && (A[1].SequenceEqual(g[1])) && (A[2].SequenceEqual(g[2])))
                return "g";

            int[][] h = new int[][] { new int[]{ A[0][0], 1, A[0][2] }, 
                                      new int[] { 1, A[1][1], 0 }, 
                                      new int[] { A[2][0], 0, 0 } };
            if ((A[0].SequenceEqual(h[0])) && (A[1].SequenceEqual(h[1])) && (A[2].SequenceEqual(h[2])))
                return "h";

            int[][] i = new int[][] { new int[] { 0, 0, A[0][2] }, 
                                    new int[]{ 0, A[1][1], 1 }, 
                                    new int[]{ A[2][0], 1, 1 } };
            if ((A[0].SequenceEqual(i[0])) && (A[1].SequenceEqual(i[1])) && (A[2].SequenceEqual(i[2])))
                return "i";

            int[][] j = new int[][] { new int[]{ 0, 0, 0 }, 
                                   new int[] { 0, A[1][1], 1 }, 
                                    new int[]{ 0, 1, 0 } };
            if ((A[0].SequenceEqual(j[0])) && (A[0].SequenceEqual(j[1])) && (A[0].SequenceEqual(j[2])))
                return "j";

            int[][] k = new int[][] {new int[] { 0, 0, 0 }, 
                                    new int[]{ 0, A[1][1], 0 }, 
                                    new int[]{ 1, 1, 1 } };
            if ((A[0].SequenceEqual(k[0])) && (A[1].SequenceEqual(k[1])) && (A[2].SequenceEqual(k[2])))
                return "k";

            int[][] l = new int[][] { new int[]{ 1, 0, 0 }, 
                                    new int[]{ 1, A[1][1], 0 }, 
                                    new int[]{ 1, 0, 0 } };
            if ((A[0].SequenceEqual(l[0])) && (A[1].SequenceEqual(l[1])) && (A[2].SequenceEqual(l[2])))
                return "l";

            int[][] m = new int[][] { new int[]{ 1, 1, 1 }, 
                                     new int[] { 0, A[1][1], 0 }, 
                                     new int[]{ 0, 0, 0 } };
            if ((A[0].SequenceEqual(m[0])) && (A[1].SequenceEqual(m[1])) && (A[2].SequenceEqual(m[2])))
                return "m";

            int[][] n = new int[][] {new int[] { 0, 0, 1 }, 
                                     new int[] { 0, A[1][1], 1 }, 
                                    new int[]{ 0, 0, 1 } };
            if ((A[0].SequenceEqual(n[0])) && (A[1].SequenceEqual(n[1])) && (A[2].SequenceEqual(n[2])))
                return "n";
            return null;
        }

        private bool[,] IsConcaveCorner(int[][] A, string Pattern)
        {
            bool[,] C = new bool[3, 3];
            switch (Pattern)
            {
                case "a":
                    {
                        if (A[0][2] == 1) C[0, 1] = true;
                        if (A[2][2] == 1) C[2, 1] = true;
                        break;
                    }
                case "b":
                    {
                        if (A[2][2] == 1) C[1, 2] = true;
                        if (A[2][0] == 1) C[1, 0] = true;
                        break;
                    }
                case "c":
                    {
                        if (A[2][0] == 1) C[2, 1] = true;
                        if (A[0][0] == 1) C[1, 2] = true;
                        break;
                    }
                case "d":
                    {
                        if (A[0][2] == 1) C[1, 2] = true;
                        if (A[0][0] == 1) C[1, 0] = true;
                        break;
                    }
                case "e":
                    {
                        if ((A[0][0] == 1) && (A[2][0] == 1)) C[1, 0] = true;
                        if ((A[2][0] == 1) && (A[2][2] == 1)) C[2, 1] = true;
                        break;
                    }
                case "f":
                    {
                        if (A[0][0] == 1) C[0, 1] = true;
                        if (A[2][0] == 1) C[1, 2] = true;
                        break;
                    }
                case "h":
                    {
                        if ((A[0][0] == 1) && (A[0][2] == 1)) C[0, 1] = true;
                        if ((A[0][0] == 1) && (A[2][0] == 1)) C[1, 0] = true;
                        break;
                    }
                case "i":
                    {
                        if (A[0][2] == 1) C[1, 2] = true;
                        if (A[2][0] == 1) C[2, 1] = true;
                        break;
                    }
            }
            return C;
        }

        private bool IsComparedO(int[][] A)
        {
            int[][] O = new int[][] { new int[]{ A[0][0], 1, A[0][2], 1 }, 
                                      new int[]{ 1, A[1][1], 1, 1 }, 
                                      new int[]{ A[2][0], 1, A[2][2], 1 }, 
                                      new int[]{ 1, 1, 1, 0 } };
            if ((A[0].SequenceEqual(O[0])) && (A[1].SequenceEqual(O[1])) && (A[2].SequenceEqual(O[2])) && (A[3].SequenceEqual(O[3])))
                return true;
            return false;
        }

        public int[,] OPATA(int[,] A, int n, int m)
        {
            bool[,] Mark = new bool[n, m];
            int[,] B = A;
            int i = 0;
            bool flag;
            do
            {
                A = B;
                flag = false;
                i = i + 1;
                for (int k = 3; k < n - 3; k++)
                {
                    for (int l = 3; l < m - 3; l++)
                    {
                        int[][] Neibourhood = new int[][] { new int []{ B[k - 1, l - 1], B[k - 1, l], B[k - 1, l + 1] }, 
                                                            new int []{ B[k, l - 1], B[k, l], B[k , l + 1] },  
                                                            new int []{ B[k + 1, l - 1], B[k + 1, l], B[k + 1, l + 1] }};

                        string Pattern = IsComparedAN(Neibourhood, B[k, l + 2], B[k + 2, l]);
                        if ((B[k, l] == 1) && (Pattern != null))
                        {
                            B[k, l] = 0;
                            flag = true;
                            bool[,] IsConcave = IsConcaveCorner(Neibourhood, Pattern);

                            for (int u = 0; u < 4; u++)
                            {
                                int col = k + offsets[u].XOffset;
                                int row = l + offsets[u].YOffset;
                                int x = 1 + offsets[u].XOffset;
                                int y = 1 + offsets[u].YOffset;

                                if (IsConcave[x, y])
                                {
                                    int[][] NeibourhoodForO = new int[][]{ new int[]{ B[col - 2, row - 2], B[col - 2, row - 1], B[col - 2, row], B[col - 2, row + 1] },                               
                                                                           new int[]{ B[col - 1, row - 2], B[col - 1, row - 1], B[col - 1, row], B[col - 1, row + 1] }, 
                                                                           new int[]{ B[col, row - 2], B[col, row - 1], B[col, row], B[col , row + 1] },  
                                                                           new int[]{ B[col + 1, row - 2], B[col + 1, row - 1], B[col + 1, row], B[col + 1, row + 1] },};

                                    if (!Mark[col, row]) Mark[col, row] = true;
                                    else
                                        if (IsComparedO(NeibourhoodForO))
                                        {
                                            B[col, row] = 0;
                                        }
                                }
                            }
                        }
                    }
                }
            }
            while (flag);
            return A;
        }
    }
}

