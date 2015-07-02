using System;
using System.Collections.Generic;
using System.Configuration;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.ImageBinarialization
{
    public class ImageBinarialization
    {
        public Bitmap BitmapImage;

        public ImageBinarialization(Bitmap src)
        {
            BitmapImage = src;
        }
        public  Bitmap Binarizator(ImageBinarialization src, int line)
        {
            Bitmap bmp = new Bitmap(src.BitmapImage);
            for (int i = 0; i < bmp.Width; i++)
            {
                for (int j = 0; j < bmp.Height; j++)
                {
                    bmp.SetPixel(i, j, bmp.GetPixel(i, j).B < line ? Color.Black : Color.White);
                }
            }
            return bmp;
        }
    }
}
