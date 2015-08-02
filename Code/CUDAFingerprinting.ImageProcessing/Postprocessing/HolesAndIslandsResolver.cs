namespace CUDAFingerprinting.ImageProcessing.Postprocessing
{
    public static class HolesAndIslandsResolver
    {
        public static int BLACK = 0;
        public static int WHITE = 255;

        private static int GetPixel(int[] data, int x, int y, int width, int height)
        {
            return (x < 0 || y < 0 || x >= width || y >= height) ?
                -1 :
                data[(height - 1 - y) * width + x];
        }

        private static int GetArea(int[] area, int x, int y, int width, int height)
        {
            return (x < 0 || y < 0 || x >= width || y >= height) ?
                -1 :
                area[(height - 1 - y) * width + x];
        }

        private static int NumberOfAreas = 0;
        private static int[] Areas;//marking
        private static int[] AreasSize;
        private static int[] Allies;//value points on prev allie

        private static void Preprocessing(int[] data, int width, int height)
        {
            NumberOfAreas = 0;

            Areas = new int[width * height];
            AreasSize = new int[width * height];
            Allies = new int[width * height];

            Allies[0] = 0;
            AreasSize[0] = 0;
            Areas[0] = 0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int leftPixel = GetPixel(data, x - 1, y, width, height);
                    int topPixel = GetPixel(data, x, y - 1, width, height);
                    int currentPixel = GetPixel(data, x, y, width, height);

                    if (leftPixel == topPixel)
                    {
                        if (topPixel != currentPixel)
                        {
                            Allies[NumberOfAreas] = NumberOfAreas;
                            AreasSize[NumberOfAreas] = 1;
                            Areas[(height - 1 - y) * width + x] = NumberOfAreas;
                            NumberOfAreas++;
                        }
                        else
                        {
                            int leftArea = GetArea(Areas, x - 1, y, width, height);
                            int topArea = GetArea(Areas, x, y - 1, width, height);

                            if (topArea != leftArea)
                            {
                                int tAR = GetAreaRoot(topArea);
                                int lAR = GetAreaRoot(leftArea);
                                Allies[lAR < tAR ? tAR : lAR] = lAR < tAR ? lAR : tAR;
                            }

                            AreasSize[leftArea]++;
                            Areas[(height - 1 - y) * width + x] = leftArea;
                        }
                    }
                    else
                    {
                        if (topPixel == currentPixel)
                        {
                            int topArea = GetArea(Areas, x, y - 1, width, height);
                            AreasSize[topArea]++;
                            Areas[(height - 1 - y) * width + x] = topArea;
                        }
                        else
                        {
                            int leftArea = GetArea(Areas, x - 1, y, width, height);
                            if (leftArea != -1)
                            {
                                AreasSize[leftArea]++;
                                Areas[(height - 1 - y) * width + x] = leftArea;
                            }
                            else
                            {
                                Allies[NumberOfAreas] = NumberOfAreas;
                                AreasSize[NumberOfAreas] = 1;
                                Areas[(height - 1 - y) * width + x] = NumberOfAreas;
                                NumberOfAreas++;
                            }
                        }
                    }
                }
            }
        }

        private static int GetAreaRoot(int area)
        {
            if (Allies[area] == area)
            {
                return area;
            }
            else
            {
                return GetAreaRoot(Allies[area]);
            }
        }

        private static int GetAreaSize(int area)
        {
            int sum = 0;
            int root = GetAreaRoot(area);
            for (int i = 0; i < NumberOfAreas; i++)
            {
                if (GetAreaRoot(i) == root)
                {
                    sum += AreasSize[i];
                }
            }
            return sum;
        }

        public static int[] ResolveHoles(int[] data, int threshold,
            int width, int height)
        {
            int[] result = (int[]) data.Clone();
            Preprocessing(data, width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (GetPixel(result, x, y, width, height) == WHITE &&
                        GetAreaSize(
                            GetArea(Areas, x, y, width, height)
                        ) < threshold)
                    {
                        result[(height - 1 - y) * width + x] = BLACK;
                    }
                }
            }
            return result;
        }

        public static int[] ResolveIslands(int[] data, int threshold,
            int width, int height)
        {
            int[] result = (int[])data.Clone();
            Preprocessing(data, width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (GetPixel(result, x, y, width, height) == BLACK &&
                        GetAreaSize(
                            GetArea(Areas, x, y, width, height)
                        ) < threshold)
                    {
                        result[(height - 1 - y) * width + x] = WHITE;
                    }
                }
            }
            return result;
        }

        public static int[] ResolveHolesAndIslands(int[] data,
             int thresholdHoles,
             int thresholdIslands,
            int width, int height)
        {
            int[] result = (int[])data.Clone();
            Preprocessing(data, width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (GetPixel(result, x, y, width, height) == BLACK &&
                        GetAreaSize(
                            GetArea(Areas, x, y, width, height)
                        ) < thresholdIslands)
                    {
                        result[(height - 1 - y) * width + x] = WHITE;
                    }
                    else if (GetPixel(result, x, y, width, height) == WHITE &&
                        GetAreaSize(
                            GetArea(Areas, x, y, width, height)
                        ) < thresholdHoles)
                    {
                        result[(height - 1 - y) * width + x] = BLACK;
                    }
                }
            }
            return result;
        }
        //JOKING :D (same algorithm, starting from an opposite corner)
        /*
        private static void PreprocessingJOKE(int[] data, int width, int height)
        {
            NumberOfAreas = 0;

            Areas = new int[width * height];
            AreasSize = new int[width * height];
            Allies = new int[width * height];

            Allies[0] = 0;
            AreasSize[0] = 0;
            Areas[0] = 0;

            for (int y = height - 1; y >= 0; y--)
            {
                for (int x = width - 1; x >= 0; x--)
                {
                    int leftPixel = GetPixel(data, x + 1, y, width, height);
                    int topPixel = GetPixel(data, x, y + 1, width, height);
                    int currentPixel = GetPixel(data, x, y, width, height);

                    if (leftPixel == topPixel)
                    {
                        if (topPixel != currentPixel)
                        {
                            Allies[NumberOfAreas] = NumberOfAreas;
                            AreasSize[NumberOfAreas] = 1;
                            Areas[(height - 1 - y) * width + x] = NumberOfAreas;
                            NumberOfAreas++;
                        }
                        else
                        {
                            int leftArea = GetArea(Areas, x + 1, y, width, height);
                            int topArea = GetArea(Areas, x, y + 1, width, height);

                            if (topArea != leftArea)
                            {
                                int tAR = GetAreaRoot(topArea);
                                int lAR = GetAreaRoot(leftArea);
                                Allies[lAR < tAR ? tAR : lAR] = lAR < tAR ? lAR : tAR;
                            }

                            AreasSize[leftArea]++;
                            Areas[(height - 1 - y) * width + x] = leftArea;
                        }
                    }
                    else
                    {
                        if (topPixel == currentPixel)
                        {
                            int topArea = GetArea(Areas, x, y + 1, width, height);
                            AreasSize[topArea]++;
                            Areas[(height - 1 - y) * width + x] = topArea;
                        }
                        else
                        {
                            int leftArea = GetArea(Areas, x + 1, y, width, height);
                            if (leftArea != -1)
                            {
                                AreasSize[leftArea]++;
                                Areas[(height - 1 - y) * width + x] = leftArea;
                            }
                            else
                            {
                                Allies[NumberOfAreas] = NumberOfAreas;
                                AreasSize[NumberOfAreas] = 1;
                                Areas[(height - 1 - y) * width + x] = NumberOfAreas;
                                NumberOfAreas++;
                            }
                        }
                    }
                }
            }
        }


        public static int[] ResolveHolesAndIslandsJOKE(int[] data,
             int thresholdHoles,
             int thresholdIslands,
            int width, int height)
        {
            int[] result = (int[])data.Clone();
            PreprocessingJOKE(data, width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (GetPixel(result, x, y, width, height) == BLACK &&
                        GetAreaSize(
                            GetArea(Areas, x, y, width, height)
                        ) < thresholdIslands)
                    {
                        result[(height - 1 - y) * width + x] = WHITE;
                    }
                    else if (GetPixel(result, x, y, width, height) == WHITE &&
                        GetAreaSize(
                            GetArea(Areas, x, y, width, height)
                        ) < thresholdHoles)
                    {
                        result[(height - 1 - y) * width + x] = BLACK;
                    }
                }
            }
            return result;
        }
        */
    }
}
