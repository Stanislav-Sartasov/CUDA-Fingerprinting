using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.RidgeLine
{
    class RidgeLine
    {
        private int[,] _image;
        private OrientationField _orientation;
        private int _step;
        private int _wings;
        private const int BuildUp = 1000;  //specially for trade coordinates between methods

        public int[] _section;

        public RidgeLine(int[,] image, int step, int wings)
        {
            _image = image;
            _orientation = new OrientationField(image);
            _step = step;
            _wings = wings;
            _section = new int[2*wings + 1];
        }

        void NewSelection(int mid)
        {
            int i = mid/BuildUp;
            int j = mid%BuildUp;

            double angle = _orientation.GetOrientation(i, j) + Math.PI / 2;

            _section[_wings] = i * BuildUp + j;

            for (var k = 1; k <= _wings; k++)
            {
                int xs, ys, xe, ye;
                xs = Convert.ToInt32(i - k * Math.Cos(angle));
                ys = Convert.ToInt32(j - k * Math.Sin(angle));
                xe = Convert.ToInt32(i + k * Math.Cos(angle));
                ye = Convert.ToInt32(j + k * Math.Sin(angle));

                _section[_wings - k] = xs * BuildUp + ys;
                _section[_wings + k] = xe * BuildUp + ye;
            }
        }

        int FindNextMax()
        {
            int x = _section[_wings] / BuildUp;
            int y = _section[_wings] % BuildUp;

            int min = _image[x, y];

            int ml = 0, mr = 0;

            for (int i = 1; i < _wings; i++)
            {
                x = _section[_wings + i] / BuildUp;
                y = _section[_wings + i] % BuildUp;

                if (_image[x, y] < min)
                {
                    mr = i;
                }

                x = _section[_wings - i] / BuildUp;
                y = _section[_wings - i] % BuildUp;

                if (_image[x, y] < min)
                {
                    ml = i;
                }
            }

            return _section[_wings + (mr < ml ? mr : -ml)];
        }

        int MakeStep(int startPoint)
        {
            int x = startPoint/BuildUp;
            int y = startPoint%BuildUp;

            double angle = _orientation.GetOrientation(x, y);

            x += Convert.ToInt32(_step * Math.Cos(angle));
            y += Convert.ToInt32(_step * Math.Sin(angle));

            return x*BuildUp + y;
        }

        bool CheckofStopCriteria()
        {
            return true;
        }

        public void GoToLine()
        {
            NewSelection(210180);

            for (int i = 0; i < _wings * 2 + 1; i++)
            {
                Console.Write("{0} ", _section[i]);
            }
            Console.WriteLine();
            for (int i = 0; i < _wings * 2 + 1; i++)
            {
                int x = _section[i] / 1000;
                int y = _section[i] % 1000;

                Console.Write("{0} ", _image[x, y]);
            }

            int max = FindNextMax();

            Console.WriteLine("\n{0}\n", max);

            NewSelection(MakeStep(max));

            for (int i = 0; i < _wings * 2 + 1; i++)
            {
                Console.Write("{0} ", _section[i]);
            }
            Console.WriteLine();
            for (int i = 0; i < _wings * 2 + 1; i++)
            {
                int x = _section[i] / 1000;
                int y = _section[i] % 1000;

                Console.Write("{0} ", _image[x, y]);
            }
            max = FindNextMax();

            Console.WriteLine("\n{0}\n", max);
        }
    }
}
