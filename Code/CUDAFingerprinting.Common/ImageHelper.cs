using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Numerics;

namespace CUDAFingerprinting.Common
{
    public static class ImageHelper
    {
        [Obsolete]
        public static void MarkMinutiae(Bitmap source, List<Minutia> minutiae, string path)
        {
            var bmp2 = new Bitmap(source.Width, source.Height);
            for (int x = 0; x < bmp2.Width; x++)
            {
                for (int y = 0; y < bmp2.Height; y++)
                {
                    bmp2.SetPixel(x, y, source.GetPixel(x, y));
                }
            }
            var gfx = Graphics.FromImage(bmp2);

            foreach (var pt in minutiae)
            {
                gfx.DrawEllipse(Pens.Red, pt.X - 2, pt.Y - 2, 5, 5);
                gfx.FillEllipse(Brushes.Red, pt.X - 2, pt.Y - 2, 5, 5);
            }

            gfx.Save();

            bmp2.Save(path, ImageFormat.Png);   
        }

        [Obsolete]
        public static void MarkMinutiae(string sourcePath, List<Minutia> minutiae, string path)
        {
            MarkMinutiae(new Bitmap(sourcePath), minutiae, path);
        }

        [Obsolete]
        public static void MarkMinutiaeWithDirections(Bitmap source, List<Minutia> minutiae, string path)
        {
            var bmp2 = new Bitmap(source.Width, source.Height);
            for (int x = 0; x < bmp2.Width; x++)
            {
                for (int y = 0; y < bmp2.Height; y++)
                {
                    bmp2.SetPixel(x, y, source.GetPixel(x, y));
                }
            }
            var gfx = Graphics.FromImage(bmp2);

            foreach (var pt in minutiae)
            {
                gfx.DrawEllipse(Pens.Red, pt.X - 2, pt.Y - 2, 5, 5);
                gfx.FillEllipse(Brushes.Red, pt.X - 2, pt.Y - 2, 5, 5);
                gfx.DrawLine(Pens.Blue, (float)pt.X, (float)pt.Y, (float)(pt.X + Math.Cos(pt.Angle)*6), (float)(pt.Y - Math.Sin(pt.Angle)*6));
            }

            gfx.Save();

            bmp2.Save(path, ImageFormat.Png);
        }

        [Obsolete]
        public static void MarkMinutiaeWithDirections(string sourcePath, List<Minutia> minutiae, string path)
        {
            MarkMinutiaeWithDirections(new Bitmap(sourcePath), minutiae, path);
        }

        [Obsolete]
        public static void MarkMinutiae(string sourcePath, List<Minutia> minutiae, List<Minutia> minutiae2, string path)
        {
            var bmp = new Bitmap(sourcePath);
            var bmp2 = new Bitmap(bmp.Width, bmp.Height);
            for (int x = 0; x < bmp2.Width; x++)
            {
                for (int y = 0; y < bmp2.Height; y++)
                {
                    bmp2.SetPixel(x, y, bmp.GetPixel(x, y));
                }
            }
            var gfx = Graphics.FromImage(bmp2);

            foreach (var pt in minutiae)
            {
                gfx.DrawEllipse(Pens.Red, pt.X - 2, pt.Y - 2, 5, 5);
                gfx.FillEllipse(Brushes.Red, pt.X - 2, pt.Y - 2, 5, 5);
            }

            foreach (var pt in minutiae2)
            {
                gfx.DrawEllipse(Pens.Blue, pt.X - 2, pt.Y - 2, 5, 5);
                gfx.FillEllipse(Brushes.Blue, pt.X - 2, pt.Y - 2, 5, 5);
            }

            gfx.Save();

            bmp2.Save(path, ImageFormat.Png);

        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static double[,] LoadImage(Bitmap bmp)
        {
            double[,] imgBytes = new double[bmp.Height, bmp.Width];
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    imgBytes[bmp.Height-1 - y, x] = bmp.GetPixel(x, y).R;
                }
            }
            return imgBytes;
        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static int[,] LoadImageAsInt(Bitmap bmp)
        {
            int[,] imgBytes = new int[bmp.Height, bmp.Width];
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    // the flipping of the image
                    imgBytes[bmp.Height-1 - y, x] = bmp.GetPixel(x, y).R;
                }
            }
            return imgBytes;
        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static double[,] LoadImage(string path)
        {
            return LoadImage(new Bitmap(path));
        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static int[,] LoadImageAsInt(string path)
        {
            return LoadImageAsInt(new Bitmap(path));
        }

        public static void SaveArray(int[,] data, string path)
        {
            SaveArrayToBitmap(data).Save(path);
        }

        public static Bitmap SaveArrayToBitmap(int[,] data)
        {
            int x = data.GetLength(1);
            int y = data.GetLength(0);
            var bmp = new Bitmap(x, y);
            data.Select2D((value, row, column) =>
            {
                value = Math.Abs(value);
                value = (value < 0) ? 0 : (value > 255 ? 255 : value);
                // note: notice the flipping
                lock(bmp)bmp.SetPixel(column, y-1-row, Color.FromArgb(value, value, value));
                return value;
            });
            return bmp;
        }

        public static Bitmap SaveArrayToBitmap(double[,] data)
        {
            int x = data.GetLength(1);
            int y = data.GetLength(0);
            var max = double.NegativeInfinity;
            var min = double.PositiveInfinity;
            foreach (var num in data)
            {
                if (num > max) max = num;
                if (num < min) min = num;
            }
            var bmp = new Bitmap(x, y);
            data.Select2D((value, row, column) =>
            {
                var gray = (int)((value - min) / (max - min) * 255);
                lock(bmp)
                    bmp.SetPixel(column, bmp.Height - 1 - row, Color.FromArgb(gray, gray, gray));
                return value;
            });
            return bmp;  
        }

        public static void SaveArrayAndOpen(double[,] data, string path)
        {
            SaveArray(data, path);
            Process.Start(path);
        }

        public static void SaveArray(double[,] data, string path)
        {
            SaveArrayToBitmap(data).Save(path);
        }

        public static ImageFormat GetImageFormatFromExtension(string path)
        {
            var extension = Path.GetExtension(path).ToUpper();
            switch(extension)
            {
                case ".BMP": return ImageFormat.Bmp;
                case ".JPG":
                case ".JPEG": return ImageFormat.Jpeg;
                case ".PNG": return ImageFormat.Png;
                default: throw new NotSupportedException();
            }
        }
    }
}
