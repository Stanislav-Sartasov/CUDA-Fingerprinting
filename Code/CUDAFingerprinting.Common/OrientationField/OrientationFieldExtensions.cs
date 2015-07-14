using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Common
{
    public static class OrientationFieldExtensions
    {
        public static void SaveToFile(this OrientationField field, string name, bool openFileAfterSaving = false)
        {
            var bmp = SaveToBitmap(field);
            bmp.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            if(openFileAfterSaving)
                Process.Start(name);
        }

        public static Bitmap SaveToBitmap(this OrientationField field)
        {
            using (var bmp = new Bitmap(field.Blocks.GetLength(1) * field.BlockSize, field.Blocks.GetLength(0) * field.BlockSize))
            {
                var gfx = Graphics.FromImage(bmp);
                gfx.FillRectangle(Brushes.White, new Rectangle(0, 0, bmp.Width, bmp.Height));
                gfx.Save();
                return SaveAboveToBitmap(field, bmp);
            }
        }
        public static Bitmap SaveAboveToBitmap(this OrientationField field, Bitmap undercoat)
        {
            var size = field.BlockSize;
            int lineLength = field.BlockSize / 2;
            var bmp = new Bitmap(undercoat.Width, undercoat.Height);
            
            for(int x=0;x<bmp.Width;x++)
            {
                for(int y=0;y<bmp.Height;y++)
                {
                    bmp.SetPixel(x, y, undercoat.GetPixel(x, y));
                }
            }

            var gfx = Graphics.FromImage(bmp);

            var pen = new Pen(Brushes.Red) { Width = 2 };
            field.Blocks.Select2D(
                (value, row, column) =>
                {
                    int x = column * size + size / 2;
                    int y = row * size + size / 2;

                    Point p0 = new Point
                    {
                        X = Convert.ToInt32(x - lineLength * Math.Cos(value.Orientation)),
                        Y = undercoat.Height-1-Convert.ToInt32(y - lineLength * Math.Sin(value.Orientation))
                    };

                    Point p1 = new Point
                    {
                        X = Convert.ToInt32(x + lineLength * Math.Cos(value.Orientation)),
                        Y = undercoat.Height - 1 - Convert.ToInt32(y + lineLength * Math.Sin(value.Orientation))
                    };

                    gfx.DrawLine(pen, p0, p1);
                    return 0;
                });
            gfx.Save();

            return bmp;
        }


		public static Bitmap SaveAboveToBitmap(this PixelwiseOrientationField field, Bitmap undercoat)
        {
            var size = field.BlockSize;
            int lineLength = field.BlockSize / 2;
            var bmp = new Bitmap(undercoat.Width * size, undercoat.Height * size);
            
            for(int x=0;x<bmp.Width;x++)
            {
                for(int y=0;y<bmp.Height;y++)
                {
                    bmp.SetPixel(x, y, undercoat.GetPixel(x / size, y / size));
                }
            }

            var gfx = Graphics.FromImage(bmp);

            var pen = new Pen(Brushes.Red) { Width = 2 };
            field.Orientation.Select2D(
                (value, row, column) =>
                {
                    int x = column * size + size / 2;
                    int y = row * size + size / 2;

                    Point p0 = new Point
                    {
                        X = Convert.ToInt32(x - lineLength * Math.Cos(value)),
                        Y = bmp.Height - 1 - Convert.ToInt32(y - lineLength * Math.Sin(value))
                    };

                    Point p1 = new Point
                    {
                        X = Convert.ToInt32(x + lineLength * Math.Cos(value)),
                        Y = bmp.Height - 1 - Convert.ToInt32(y + lineLength * Math.Sin(value))
                    };

                    gfx.DrawLine(pen, p0, p1);
                    return 0;
                });
            gfx.Save();

            return bmp;
        }

        public static void SaveAboveToFile(this OrientationField field, Bitmap undercoat, string name, bool openFileAfterSaving = false)
        {
            var bmp = SaveAboveToBitmap(field, undercoat);
            bmp.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            if (openFileAfterSaving)
                Process.Start(name);
        }


		public static void SaveAboveToFile(this PixelwiseOrientationField field, Bitmap undercoat, string name, bool openFileAfterSaving = false)
		{
			var bmp = SaveAboveToBitmap(field, undercoat);
			bmp.Save(name, ImageHelper.GetImageFormatFromExtension(name));
			if (openFileAfterSaving)
				Process.Start(name);
		}
    }
}
