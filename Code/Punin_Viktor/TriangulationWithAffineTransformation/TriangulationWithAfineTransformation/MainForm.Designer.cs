namespace TriangulationWithAfineTransformation
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.ImageFrom = new System.Windows.Forms.PictureBox();
            this.ImageTo = new System.Windows.Forms.PictureBox();
            this.Match = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.ImageFrom)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ImageTo)).BeginInit();
            this.SuspendLayout();
            // 
            // ImageFrom
            // 
            this.ImageFrom.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.ImageFrom.Location = new System.Drawing.Point(12, 12);
            this.ImageFrom.Name = "ImageFrom";
            this.ImageFrom.Size = new System.Drawing.Size(229, 274);
            this.ImageFrom.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.ImageFrom.TabIndex = 0;
            this.ImageFrom.TabStop = false;
            this.ImageFrom.Click += new System.EventHandler(this.ImageFromOnClick);
            // 
            // ImageTo
            // 
            this.ImageTo.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.ImageTo.Location = new System.Drawing.Point(394, 12);
            this.ImageTo.Name = "ImageTo";
            this.ImageTo.Size = new System.Drawing.Size(229, 274);
            this.ImageTo.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.ImageTo.TabIndex = 1;
            this.ImageTo.TabStop = false;
            this.ImageTo.Click += new System.EventHandler(this.ImageToOnClick);
            // 
            // Match
            // 
            this.Match.Location = new System.Drawing.Point(282, 12);
            this.Match.Name = "Match";
            this.Match.Size = new System.Drawing.Size(75, 23);
            this.Match.TabIndex = 2;
            this.Match.Text = "Match";
            this.Match.UseVisualStyleBackColor = true;
            this.Match.Click += new System.EventHandler(this.MatchButtonClick);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoSize = true;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(635, 300);
            this.Controls.Add(this.Match);
            this.Controls.Add(this.ImageTo);
            this.Controls.Add(this.ImageFrom);
            this.Name = "MainForm";
            this.Text = "Triangulation with afine transformation";
            this.ResizeEnd += new System.EventHandler(this.OnMainFormResize);
            this.Resize += new System.EventHandler(this.OnMainFormResize);
            ((System.ComponentModel.ISupportInitialize)(this.ImageFrom)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ImageTo)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox ImageFrom;
        private System.Windows.Forms.PictureBox ImageTo;
        private System.Windows.Forms.Button Match;
    }
}

