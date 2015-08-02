using System.Linq;

namespace CUDAFingerprinting.Common
{
    public static class MorphologicalOperators
    {
        public static int BLACK = 0;
        public static int WHITE = 255;

        private static int GetPixel(int[] data, int x, int y, int width, int height)
        {
            return (x < 0 || y < 0 || x >= width || y >= height) ?
                WHITE :
                data[(height - 1 - y) * width + x];
        }

        private static bool CheckErosion(int[] data, int[] structEl,
            int x, int y,
            int width, int height,
            int sWidth, int sHeight)
        {
            if (GetPixel(data, x, y, width, height) != BLACK)
                return false;
            for (int dy = -sHeight / 2; dy < sHeight / 2 + sHeight % 2; dy++)
            {
                for (int dx = -sWidth / 2; dx < sWidth / 2 + sWidth % 2; dx++)
                {
                    int pix = GetPixel(structEl, dx + sWidth / 2, dy + sHeight / 2, sWidth, sHeight);
                    if (pix == BLACK &&
                        pix != GetPixel(data, x + dx, y + dy, width, height))
                        return false;
                }
            }
            return true;
        }

        public static int[] Erosion(int[] data, int[] structEl,
            int width, int height,
            int sWidth, int sHeight)
        {
            int[] result = Enumerable.Repeat(WHITE, data.Length).ToArray();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (CheckErosion(data, structEl,
                        x, y,
                        width, height,
                        sWidth, sHeight))
                    {
                        result[(height - 1 - y) * width + x] = BLACK;
                    }
                }
            }
            return result;
        }

        private static bool CheckDilation(int[] data, int[] structEl,
            int x, int y,
            int width, int height,
            int sWidth, int sHeight)
        {
            for (int dy = -sHeight / 2; dy < sHeight / 2 + sHeight % 2; dy++)
            {
                for (int dx = -sWidth / 2; dx < sWidth / 2 + sWidth % 2; dx++)
                {
                    int pix = GetPixel(structEl, dx + sWidth / 2, dy + sHeight / 2, sWidth, sHeight);
                    if (pix == BLACK &&
                        pix == GetPixel(data, x + dx, y + dy, width, height))
                        return true;
                }
            }
            return false;
        }

        public static int[] Dilation(int[] data, int[] structEl,
            int width, int height,
            int sWidth, int sHeight)
        {
            int[] result = Enumerable.Repeat(WHITE, data.Length).ToArray();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (CheckDilation(data, structEl,
                        x, y,
                        width, height,
                        sWidth, sHeight))
                    {
                        result[(height - 1 - y) * width + x] = BLACK;
                    }
                }
            }
            return result;
        }

        public static int[] Opening(int[] data, int[] structEl,
            int width, int height,
            int sWidth, int sHeight)
        {
            return Dilation(
                Erosion(
                    data,
                    structEl,
                    width, height,
                    sWidth, sHeight
                ),
                structEl,
                width, height,
                sWidth, sHeight);
        }

        public static int[] Closing(int[] data, int[] structEl,
            int width, int height,
            int sWidth, int sHeight)
        {
            return Erosion(
                Dilation(
                    data,
                    structEl,
                    width, height,
                    sWidth, sHeight
                ),
                structEl,
                width, height,
                sWidth, sHeight);
        }
    }
}
