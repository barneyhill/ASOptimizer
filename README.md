# ASOptimizer
> **ASOptimizer consists of a database and two computational models: *sequence engineering* and *chemical engineering***.

## 0. Requirements
- Python 3.6
- Install required libraries using the following command:

> <pre><code>pip3 install -r requirements.txt</code></pre>

- Please download the following data files and place npy files into the "/sequence_engineering/features" folder
  - https://drive.google.com/file/d/10sRArsEHu6S5W0sbC-Vt4iLCMfzaiUG6/view?usp=sharing

- Please download the following data files and place csv files into the "/chemical_engineering/top6" folder
  - https://drive.google.com/file/d/1UfcV5JvG4JdeOJn1kpKokuhfua4hptjz/view?usp=sharing

## 1. Database

### 1.1 For sequence engineering
- **/ido-patent/EFO21_experiments.csv**, **/ido-patent/SK0V3_experiments.csv**
  - Database of experimental observations from the granted patent "Immunosuppression-reverting oligonucleotides inhibiting the expression of IDO"

- **/patent_experiments/aso_features_patent.csv**
  - ASO candidates from the granted patent "Immunosuppression-reverting oligonucleotides inhibiting the expression of IDO" and corresponding features
    
- **/features/IDO1_features.csv**
  - 19-mer ASO candidates for regulating IDO1 mRNA and corresponding features
   
### 1.2 For chemical engineering
- **/baseline/predictions_sequence_information.csv**
  - Our predictions and experimental ground truth (actual experimental results) for HIF1A experiments

- **/insilico/predictions_sequence_information.csv**
  - Our predictions and experimental ground truth (actual experimental results) for our test dataset from literature

- **/top6/predictions_sequence_information.csv**
  - Our predictions on the top 6 sequences for regulating IDO1 mRNA


## 2. Sequence Engineering

To perform sequence engineering, use the following command:
> <pre><code>python3 main.py --mode 'train' --target 'ido1' --seq_len 19 --num_candidates 6 --rnastructure 'mfold' </code></pre>

To perform sequence engineering with *your chosen setting*, use the following command:
> <pre><code>python3 main.py --mode 'eval' --target 'ido1' --seq_len 19 --num_candidates 6 --rnastructure 'mfold' --a_star "1 1 1 1" </code></pre>

### Code

- **Linear_regression.ipynb**: 
  - Displays a scatter plot comparing experimentally observed inhibition rates (x-axis) with their predicted values (y-axis) (Figure 3A and Figure 3B)
  - Comparison with Sfold

- **Plot_Contour.ipynb**: 
  - Displays a surface plot of Pearson correlation (ρ), represented by contour plots (Figure 3C and Figure 3D)

- **Getting_ASOpt_top6.ipynb**: 
  - Shows a histogram depicting the predicted scores of complementary ASOs, each 19 nucleotides in length, targeting the IDO1 gene (Figure 4)

## 3. Chemical Engineering

### Code

- **comparison_to_[13].ipynb**: 
  - Performance comparison with baseline [13] (Figure S2)

- **ASOptimizer.ipynb**: 
  - Presents in silico validation results for the chemical engineering module (Figure S3)
  - Displays a histogram of ASOptimizer scores for the top six sequences targeting IDO regulation, considering all possible chemical combinations (Figure S4)
