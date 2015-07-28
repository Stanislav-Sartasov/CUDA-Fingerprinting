﻿using System;
using System.Collections.Generic;
using System.Drawing.Design;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.RidgeLine
{
    class RidgeLine
    {
        private int[,] _image;
        private OrientationField _orientation;
        private int _step;
        private int _wings;
        private bool _diffAngle;
        private bool[,] _visited;
        private const int BuildUp = 1000;  //specially for trade coordinates between methods
        private const double _pi4 = Math.PI/4;

        public int[] _section;

        public RidgeLine(int[,] image, int step, int wings)
        {
            _image = image;
            _orientation = new OrientationField(image);
            _step = step;
            _wings = wings;
            _section = new int[2*wings + 1];
            _diffAngle = false;
            _visited = new bool[image.GetLength(0), image.GetLength(1)];
        }

        void NewSection(int mid)
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
        } //All right

        int[] FindEdges()
        {
            int lPoint = _wings;
            int rPoint = _wings;

            bool check = false;

            while (!check)
            {
                int x = _section[lPoint] / BuildUp;
                int y = _section[lPoint] % BuildUp;

                if (_image[x, y] > 125)
                {
                    check = true;
                }
                else
                {
                    lPoint--;

                    if (lPoint < 0) break;
                }
            }

            while (check)
            {
                int x = _section[rPoint] / BuildUp;
                int y = _section[rPoint] % BuildUp;

                if (_image[x, y] > 125)
                {
                    check = false;
                }
                else
                {
                    rPoint--;

                    if (rPoint > _wings * 2) break;
                }
            }

            var res = new int[2] {lPoint, rPoint};

            return res;
        }  

        private int MakeStep(int startPoint)
        {
            int x = startPoint/BuildUp;
            int y = startPoint%BuildUp;

            double angle = _orientation.GetOrientation(x, y) + (_diffAngle ? Math.PI : 0);

            x += (int)(_step*Math.Cos(angle) + 0.5);
            y += (int)(_step*Math.Sin(angle) + 0.5);

            double angle2 = _orientation.GetOrientation(x, y);

            if (Math.Abs(angle - angle2) > _pi4)
            {
                _diffAngle = true;
            }
            else
            {
                _diffAngle = false;
            }

            return x*BuildUp + y;
        }

        void Paint(int lstartPoint, int rstartPoint, double angle, int stepDistX, int stepDistY)
        {
            int xd = (int)(Math.Cos(angle) + 0.5);
            int yd = (int)(Math.Sin(angle) + 0.5);

            for (int i = lstartPoint; i <= rstartPoint; i++)
            {
                int xcur = _section[i]/BuildUp;
                int ycur = _section[i]%BuildUp;

                int xLimit = xcur + stepDistX;
                int yLimit = ycur + stepDistY;

                while ((ycur < yLimit) && (xcur < xLimit))
                {
                    _visited[ycur, xcur] = true;
                    ycur += yd;
                    xcur += xd;
                }
            }

            //int i = ystart;
            //int j = xstart;
            //while ((i <= yend) && (j <= xend))
            //{
            //    int ycur = i;
            //    int xcur = j;
            //    int xLimit = xcur + stepDistX;
            //    int yLimit = ycur + stepDistY;
            //    while ((ycur < yLimit) && (xcur < xLimit))
            //    {
            //        _visited[xcur, ycur] = true;
            //        ycur += yd;
            //        xcur += xd;
            //    }
            //    i += ydAlongLine;
            //    j += xdAlongLine;
            //}
        }

        bool CheckStopCriteria(int threshold = 200)
        {
            double mean = 0;
            for (int i = 0; i < _wings*2 + 1; i++)
                mean += _section[i];
            mean /= (_wings*2 + 1);
            if (mean > threshold) return true;
            return false;
        }

        public void GoToLine()
        {

        }
    }
}