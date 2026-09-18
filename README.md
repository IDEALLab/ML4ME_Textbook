# Machine Learning for Mechanical Engineering

This is a textbook repository for the ETHZ Machine Learning for Mechanical Engineering course.

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IDEALLab/ML4ME_Textbook/main?urlpath=lab/tree/part1)

---

## 📚 For Students: Setup Guide

This guide provides step-by-step instructions to set up the machine learning environment required for this course. No prior Python experience is required.

### ☁️ Fastest: Run in Your Browser with Binder (No Install, No Account)

Click the badge below (or the one at the top of this page). After a short wait a JupyterLab
session opens in your browser with every package this book needs already installed.

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IDEALLab/ML4ME_Textbook/main?urlpath=lab/tree/part1)

1. **Click the badge.** The first launch after a repository update can take several minutes while
   the environment builds. Later launches usually take under a minute.
2. **Open a notebook** from the file browser on the left (start in the `part1/` folder).
3. **Run cells** with `Shift+Enter`.

**Good to know:**
- **Your work is not saved on the server.** A session shuts down after about 10 minutes of
  inactivity, and everything in it is discarded. To keep your changes, download the notebook
  (right-click the file → *Download*) before you leave. You can upload it again next time.
- Sessions are limited to roughly 1–2 GB of memory and have **no GPU**. That is plenty for Part 1
  and most exercises. For the heavier generative-model notebooks in Part 2, use the local setup
  or Colab (below).
- If a launch fails with a capacity message, wait a minute and try again. Binder is a free,
  shared service.

The environment Binder builds is defined in [`binder/requirements.txt`](binder/requirements.txt).
It is separate from the local setup below, so changing one does not affect the other.

---

### 🌟 Alternative: Google Colab (Requires a Google Account)

**For students who prefer a cloud-based environment without local installation:**

1. **Go to [colab.research.google.com](https://colab.research.google.com)**
2. **Sign in with your Google account**
3. **Click "New Notebook"**
4. **Upload any `.ipynb` file from this book** (from the `notebooks` folder)
5. **Install the few packages Colab does not ship** by running this in the first cell
   (PyTorch, NumPy, pandas, scikit-learn and matplotlib are already installed on Colab):
   ```python
   !pip install --quiet pyro-ppl torchdiffeq geomloss gymnasium engibench
   ```
6. **Enable GPU (optional):** Runtime → Change runtime type → GPU → Save

**Benefits:**
- No local installation required
- Free GPU access (CUDA automatically configured)
- Runs entirely in your browser
- Most packages pre-installed

**Note:** You will need to reinstall packages each time you start a new Colab session.

---

### 💻 Local Setup (Recommended for Course)

### 🚀 Quick Start (3 Steps)

#### Step 1: Install Prerequisites
**You need these 2 things:**

1. **Miniforge** (Python package manager)
   - Download from the official [conda-forge website](https://conda-forge.org/download/)
   - Choose the appropriate installer for your operating system
   - **Windows:** Double-click the `.exe` file and follow the installer
   - **Mac/Linux:** After downloading, run these commands in Terminal:
     ```bash
     # For Mac (choose the right one for your chip):
     chmod +x Miniforge3-latest-MacOSX-arm64.sh    # Apple Silicon (M1/M2/M3)
     # or
     chmod +x Miniforge3-latest-MacOSX-x86_64.sh   # Intel Mac
     # or
     chmod +x Miniforge3-latest-Linux-x86_64.sh    # Linux
     
     # Then run the appropriate installer:
     ./Miniforge3-latest-MacOSX-arm64.sh           # Apple Silicon
     # or
     ./Miniforge3-latest-MacOSX-x86_64.sh          # Intel Mac  
     # or
     ./Miniforge3-latest-Linux-x86_64.sh           # Linux
     ```

2. **VS Code** (code editor)
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)
   - Install it (just click through the installer)

#### Step 2: Get This Book
**Copy this book to your computer (choose one method):**

**Method A: VS Code Git Clone (Recommended)**
1. **Open VS Code**
2. **Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)**
3. **Type "Git: Clone" and press Enter**
4. **Paste this URL:** `https://github.com/IDEALLab/ML4ME_Textbook.git`
5. **Choose a folder** (like Desktop or Documents)
6. **Wait for it to download**

**Method B: Direct Download (Easiest)**
1. **Go to [github.com/IDEALLab/ML4ME_Textbook](https://github.com/IDEALLab/ML4ME_Textbook)**
2. **Click "Code" → "Download ZIP"**
3. **Extract the ZIP file** to your desired folder
4. **Open the folder in VS Code**

#### Step 3: Run the Auto-Setup
**This installs everything automatically:**

1. **Open Terminal/Command Prompt:**
   - **Windows:** Open "Miniforge Prompt" from Start Menu (NOT regular Command Prompt!)
   - **Mac:** Press `Cmd+Space`, type "Terminal", press Enter
   - **Linux:** Press `Ctrl+Alt+T`

2. **Navigate to the book folder:**
   ```bash
   cd ML4ME_Textbook
   ```

3. **Run the setup script:**
   - **Windows:** Type `python bootstrap_env.py` in Miniforge Prompt
   - **Mac/Linux:** Type `python bootstrap_env.py` in terminal

4. **Wait 5-10 minutes** (the script downloads and installs required software)

5. **Setup complete**

### 📝 Optional: Create GitHub Account

**You only need a GitHub account if you want to:**
- Contribute to the course materials
- Create your own repositories
- Use advanced Git features

**To create a free account:**
1. **Go to [github.com](https://github.com)**
2. **Click "Sign up"**
3. **Create your account** (use your ETH email if you have one)
4. **Verify your email**

**Note:** This is completely optional for just using the course materials.

### ✅ What the Setup Script Does

The setup script automatically:
- ✅ Created a Python environment called `ml4me-student`
- ✅ Installed PyTorch (for deep learning) with proper CUDA support
- ✅ Installed all ML libraries from the project dependencies (NumPy, Matplotlib, Pandas, Scikit-learn, Jupyter, etc.)
- ✅ Set up everything needed to run the interactive notebooks

### 🎯 How to Start Learning

#### Read the Textbook Online
The complete textbook is available online at: **https://ideal.umd.edu/ML4ME_Textbook/**

#### Run Interactive Notebooks in VS Code
VS Code provides an integrated development environment with excellent Jupyter support and is the recommended approach for this course.

1. **Open VS Code**
2. **Open the book folder**
3. **Click on any `.ipynb` file in the `notebooks` folder**
4. **VS Code will ask you to select a kernel - choose "ml4me-student"**
5. **Start coding!** The notebook will run directly in VS Code

**Advantages of VS Code for ML development:**
- **Integrated experience** - no switching between browser and terminal
- **Smart autocomplete** and IntelliSense for Python
- **Built-in debugging** tools for troubleshooting
- **Git integration** for version control
- **Extension ecosystem** for ML/AI development
- **Integrated terminal** for running commands



### 🔧 Daily Workflow

**Every time you want to work on this book:**

#### VS Code Workflow (Recommended):
1. **Open VS Code**
2. **Open the book folder** (File → Open Folder)
3. **Click on any `.ipynb` file** in the `notebooks` folder
4. **Select "ml4me-student" kernel** when prompted
5. **Start coding!** Everything runs in VS Code

**When you're done:**
- Just close VS Code or the notebook file - that's it!


### 🆘 Troubleshooting

#### "Conda not found"
- **Solution:** Install Miniforge from [conda-forge.org/download](https://conda-forge.org/download/)
- **Restart your terminal** after installing

#### "Permission denied" (Mac/Linux)
- **Solution:** Make sure you have Python installed and accessible in your terminal

#### "Python not found"
- **Solution:** Make sure you activated the environment: `conda activate ml4me-student`


#### VS Code can't find Python
- **Solution:** 
  1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
  2. Type "Python: Select Interpreter"
  3. Choose the one with `ml4me-student`

#### VS Code can't find the conda kernel
- **Solution:**
  1. Open a `.ipynb` file in VS Code
  2. **Method 1:** Click on the kernel name in the top-right corner of the notebook
  3. **Method 2:** Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac), type "Notebook: Select Notebook Kernel"
  4. Select "ml4me-student" from the list
  5. If not listed, select "Select Another Kernel" → "Python Environments" → "ml4me-student"

#### Windows: "conda command not found" in regular Command Prompt
- **Solution:** Always use "Miniforge Prompt" from Start Menu, NOT regular Command Prompt

#### GitHub: "Authentication failed" when cloning
- **Solution:** Use Method B (Direct Download) instead, or create a free GitHub account if you want to use Git features

#### Binder: launch is stuck on "Building" or "Launching"
- **Solution:** The first launch after the book is updated rebuilds the environment, which can
  take 5–15 minutes. Leave the tab open. If it fails outright, wait a minute and click the badge again.

#### Binder: my changes are gone
- **Solution:** Binder sessions are temporary and end after ~10 minutes of inactivity. Download your
  notebook (right-click → *Download*) before leaving, and upload it into a new session to continue.
  For work you want to keep long-term, use the local setup.

#### Binder: kernel dies or "Kernel restarting" while training
- **Solution:** The session ran out of memory (about 2 GB). Reduce the dataset or model size, or
  run that notebook locally or on Colab.

#### Colab: "Package not found" error
- **Solution:** Run the package installation cell first:
  ```python
  !pip install --quiet pyro-ppl torchdiffeq geomloss gymnasium engibench
  ```

#### Colab: "CUDA out of memory" error
- **Solution:** 
  1. Runtime → Restart runtime
  2. Or use CPU: Runtime → Change runtime type → CPU → Save

#### Colab: Notebook disconnects after inactivity
- **Solution:** This is normal - just reconnect and re-run the package installation cell

### 🔗 Useful Commands

| What you want to do | Command |
|---------------------|---------|
| Activate environment | `conda activate ml4me-student` |
| Deactivate environment | `conda deactivate` |
| Check Python version | `python --version` |
| List installed packages | `pip list` |
| Update a package | `pip install --upgrade package_name` |

### Getting Help

#### If you encounter issues:
1. **Check this guide first** - most common problems are covered here
2. **Ask your classmates** - they may have encountered the same issue
3. **Contact your instructor** for additional support

#### Common Issues:
- **"Module not found"** → Make sure you selected the "ml4me-student" kernel in VS Code
- **CUDA/NVIDIA messages when activating environment** → This is normal on Windows/Linux with NVIDIA GPUs; macOS doesn't show these because CUDA packages aren't present there

---

## 👨‍🏫 For Instructors: Compiling the Book

I am using [Quarto](https://quarto.org/) for this book, and you can render the book using the following steps:

1. Install [Quarto](https://quarto.org/docs/get-started/).
2. Clone this repository.
3. Preview the book via the command line (allows you to edit the book and see changes live):
```bash
quarto preview
```
4. Render the book via the command line (allows you to compile the book into HTML in `_book`):
```bash
quarto render
```
5. If you want to render the book into PDF (will be placed in the `_book` folder), you can use:
```bash
quarto render --to pdf
```

---

## Setup Complete

You now have a complete machine learning environment set up. You can:

- Read the interactive textbook
- Run all the example notebooks
- Complete the problem sets
- Start your own ML projects

---

*This setup guide was created to make machine learning accessible to everyone, regardless of their programming background. If you have suggestions for improvements, please let us know!*