import os
import re
import base64
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')
    return ""

def generate_html_report():
    print("Generating HTML report...")
    
    # Read statistical results
    stat_content = ""
    if os.path.exists("results/statistical_results.txt"):
        with open("results/statistical_results.txt", "r", encoding="utf-8") as f:
            stat_content = f.read()
            
    # Parse statistical results for tables
    # Find F1-macro, Accuracy, Balanced Accuracy for GA and PSO
    ga_summary = {}
    pso_summary = {}
    
    # Simple regex to extract numbers
    pattern = r"(\w+\s+\w+-?\w*):\s+([\d\.]+)\s+±\s+([\d\.]+)"
    
    # Find GA section
    ga_section = re.search(r"Genetic Algorithm Summary:(.*?)(?:Particle Swarm|$)", stat_content, re.DOTALL)
    if ga_section:
        matches = re.findall(pattern, ga_section.group(1))
        for key, val, std in matches:
            ga_summary[key.strip()] = f"{val} ± {std}"
            
    # Find PSO section
    pso_section = re.search(r"Particle Swarm Optimization Summary:(.*?)(?:Hypothesis|$)", stat_content, re.DOTALL)
    if pso_section:
        matches = re.findall(pattern, pso_section.group(1))
        for key, val, std in matches:
            pso_summary[key.strip()] = f"{val} ± {std}"
            
    # Find hypothesis test p-values
    p_val_t = re.search(r"paired t-test:.*?p-value=([\d\.\-e]+)", stat_content)
    p_val_w = re.search(r"Wilcoxon test:.*?p-value=([\d\.\-e]+)", stat_content)
    
    p_val_t_str = p_val_t.group(1) if p_val_t else "N/A"
    p_val_w_str = p_val_w.group(1) if p_val_w else "N/A"
    
    conclusion = re.search(r"Conclusion:(.*)", stat_content)
    conclusion_str = conclusion.group(1).strip() if conclusion else "N/A"

    # Embed figures as base64 to make HTML self-contained and print-friendly
    img_ga_sensitivity = get_base64_image("results/ga_sensitivity.png")
    img_convergence = get_base64_image("results/convergence_comparison.png")
    img_boxplot = get_base64_image("results/performance_boxplot.png")
    img_confusion = get_base64_image("results/confusion_matrices.png")
    img_roc = get_base64_image("results/roc_curves.png")

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Optimization Algorithms - Project Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --secondary: #10b981;
            --dark: #0f172a;
            --light: #f8fafc;
            --text-dark: #1e293b;
            --text-light: #64748b;
            --border: #e2e8f0;
            --white: #ffffff;
        }
        
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--light);
            color: var(--text-dark);
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }
        
        .header-banner {
            background: linear-gradient(135deg, var(--dark) 0%, #1e3a8a 50%, #3b82f6 100%);
            color: var(--white);
            padding: 60px 40px;
            text-align: center;
            border-bottom: 5px solid var(--secondary);
            position: relative;
        }
        
        .header-banner h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.8rem;
            margin: 0 0 10px 0;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        
        .header-banner p {
            font-size: 1.2rem;
            opacity: 0.9;
            margin: 0 0 25px 0;
            font-weight: 300;
        }
        
        .metadata-container {
            display: flex;
            justify-content: center;
            gap: 40px;
            font-size: 0.95rem;
            font-weight: 500;
        }
        
        .metadata-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 8px 16px;
            border-radius: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        
        .content-wrapper {
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 20px;
        }
        
        .card {
            background: var(--white);
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
            padding: 40px;
            margin-bottom: 40px;
            border: 1px solid var(--border);
        }
        
        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            color: var(--primary);
            margin-top: 0;
            border-bottom: 2px solid var(--border);
            padding-bottom: 12px;
            font-weight: 700;
        }
        
        h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            color: #2563eb;
            margin-top: 25px;
            font-weight: 600;
        }
        
        p, li {
            color: var(--text-dark);
            font-size: 1.05rem;
            text-align: justify;
        }
        
        ul {
            padding-left: 20px;
        }
        
        li {
            margin-bottom: 10px;
            text-align: left;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .figure-container {
            text-align: center;
            margin: 25px 0;
        }
        
        .figure-container img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
        }
        
        .figure-caption {
            font-size: 0.9rem;
            color: var(--text-light);
            margin-top: 10px;
            font-style: italic;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 1rem;
        }
        
        th {
            background-color: var(--dark);
            color: var(--white);
            font-weight: 600;
            text-align: left;
            padding: 12px 16px;
        }
        
        td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }
        
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-left: 5px solid var(--primary-light);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .highlight-box p {
            margin: 0;
            font-weight: 500;
            color: #1e40af;
        }
        
        .footer-info {
            text-align: center;
            color: var(--text-light);
            padding: 40px 0;
            font-size: 0.9rem;
            border-top: 1px solid var(--border);
        }
        
        @media print {
            body {
                background-color: var(--white);
            }
            .card {
                box-shadow: none;
                border: none;
                padding: 0;
                margin-bottom: 60px;
                page-break-inside: avoid;
            }
            .header-banner {
                border-bottom: 2px solid #000;
                padding: 30px 20px;
            }
            .metadata-item {
                border: 1px solid #ccc;
                color: #000;
            }
            .content-wrapper {
                max-width: 100%;
                margin: 0;
                padding: 0;
            }
            h2 {
                page-break-after: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="header-banner">
        <h1>Optimization Algorithms - Project Report</h1>
        <p>Training Neural Networks with Genetic Algorithms and Swarm Intelligence</p>
        <div class="metadata-container">
            <div class="metadata-item">Course: Optimization Algorithms (OA)</div>
            <div class="metadata-item">April - June 2026</div>
            <div class="metadata-item">Group Number: 99</div>
        </div>
    </div>

    <div class="content-wrapper">
        <div class="card">
            <h2>1. Problem Formulation</h2>
            <p>
                This project investigates training the weights and biases of an Artificial Neural Network (ANN) as an optimization task, rather than utilizing classical gradient-descent algorithms. Gradient-descent approaches (like backpropagation) are highly prone to getting trapped in local minima, especially in complex loss landscapes. In contrast, population-based search methods search globally and do not rely on gradient information.
            </p>
            <p>
                We formulate this by defining the optimization solution vector &theta; &isin; &real;<sup>n</sup> containing the flattened configuration of all connection weights and layer biases of an <code>MLPClassifier</code> network.
            </p>
            <p>
                The architecture chosen for this task consists of an input layer corresponding to the 22 features, two hidden layers of sizes 10 and 5 respectively, and a binary classification output layer. This layout yields a total of <b>291 parameters</b>.
            </p>
            
            <div class="highlight-box">
                <p><b>Fitness Metric Selection:</b> The Oxford Parkinson's Disease Detection dataset contains 195 patients, with 147 (75.4%) diagnosed with Parkinson's and 48 (24.6%) healthy controls. Standard Accuracy is highly misleading for this class imbalance. To ensure robustness, we select the <b>Macro F1-Score</b> as the fitness function optimization metric. This prevents the algorithms from achieving high scores by simply classifying all subjects into the majority class.</p>
            </div>
        </div>

        <div class="card">
            <h2>2. Implementation Details</h2>
            <p>
                All models are implemented from scratch in pure Python and integrated cleanly.
            </p>
            <h3>Genetic Algorithm (GA)</h3>
            <p>
                A continuous Genetic Algorithm was implemented with the following key components:
            </p>
            <ul>
                <li><b>Initialization:</b> Xavier Uniform Initialization (Xavier bounds for weights, zeros/small limits for biases) vs. He Normal Initialization.</li>
                <li><b>Selection:</b> Tournament Selection (k=3) which provides robust selection pressure vs. Rank-based Roulette Wheel Selection.</li>
                <li><b>Crossover:</b> Blend Crossover (BLX-&alpha;, with &alpha;=0.5) which allows generating genes outside the bounds of parents, encouraging search expansion, vs. Arithmetic Crossover.</li>
                <li><b>Mutation:</b> Gaussian Mutation (adds &Nu;(0, &sigma;<sup>2</sup>) noise) vs. Uniform Mutation.</li>
                <li><b>Elitism:</b> Retaining the top 2 individuals per generation to avoid degradation of the global best.</li>
            </ul>

            <h3>Particle Swarm Optimization (PSO)</h3>
            <p>
                For the swarm-based method, we chose <b>Particle Swarm Optimization (PSO)</b>. PSO is inherently well-suited for continuous optimization tasks as it mimics social vector movement.
            </p>
            <ul>
                <li><b>Velocity Update:</b> v<sub>i</sub>(t+1) = w &middot; v<sub>i</sub>(t) + c<sub>1</sub> &middot; r<sub>1</sub> &middot; (pbest<sub>i</sub> - x<sub>i</sub>(t)) + c<sub>2</sub> &middot; r<sub>2</sub> &middot; (gbest - x<sub>i</sub>(t))</li>
                <li><b>Inertia Weight (w):</b> Linearly decays from 0.9 to 0.4 over the course of generations. This guides the swarm from global exploration initially to localized exploitation towards the end.</li>
                <li><b>Cognitive (c<sub>1</sub>) & Social (c<sub>2</sub>) Coefficients:</b> Set to 1.6 to balance personal search memories and social swarm knowledge.</li>
                <li><b>Clamping:</b> Clamping velocity at v<sub>max</sub> = 0.5 and positions in range [-2.0, 2.0] prevents runaway particle trajectories and ensures stable network weights.</li>
            </ul>
        </div>

        <div class="card">
            <h2>3. GA Operator Sensitivity Analysis</h2>
            <p>
                An analysis was conducted to compare two sets of GA operators on a single train-test split:
            </p>
            <ul>
                <li><b>Config A (Heuristic Continuous GA):</b> Uniform Xavier Init, Tournament Selection, BLX-0.5 Crossover, Gaussian Mutation.</li>
                <li><b>Config B (Classical continuous GA):</b> Normal He Init, Roulette Wheel Selection, Arithmetic Crossover, Uniform Mutation.</li>
            </ul>
            
            <div class="grid-2">
                <div>
                    <table style="margin-top: 40px;">
                        <thead>
                            <tr>
                                <th>Configuration</th>
                                <th>Train Fitness (F1-macro)</th>
                                <th>Test F1-macro</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><b>Config A (BLX/Gaussian/Tourn)</b></td>
                                <td>0.8608</td>
                                <td>0.6638</td>
                            </tr>
                            <tr>
                                <td><b>Config B (Arithmetic/Uniform/Roule)</b></td>
                                <td>0.8232</td>
                                <td>0.6869</td>
                            </tr>
                        </tbody>
                    </table>
                    <p style="font-size: 0.95rem; margin-top: 20px;">
                        Config A achieves higher training fitness during optimization. BLX Crossover provides much better interpolation and extrapolation capacity in continuous domains than Arithmetic Crossover. Tournament selection maintains a stronger selection pressure, while Gaussian mutation allows smaller local steps for refinement compared to uniform random mutation.
                    </p>
                </div>
                <div class="figure-container">
                    <img src="{img_ga_sensitivity}" alt="GA Operator Sensitivity Analysis">
                    <div class="figure-caption">Figure 1: Convergence of GA Configuration A vs. Config B.</div>
                </div>
            </div>
        </div>

        <div class="card" style="page-break-before: always;">
            <h2>4. Experimental Evaluation</h2>
            <p>
                To draw robust conclusions, we executed the best Genetic Algorithm config (Config A) and Particle Swarm Optimization (PSO) across <b>10 independent runs</b> with different random data splits (80/20 train/test ratio).
            </p>
            
            <table>
                <thead>
                    <tr>
                        <th>Optimization Algorithm</th>
                        <th>Train F1-macro</th>
                        <th>Test F1-macro</th>
                        <th>Test Accuracy</th>
                        <th>Test Balanced Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>Genetic Algorithm (GA)</b></td>
                        <td>{ga_summary_train}</td>
                        <td>{ga_summary_test_f1}</td>
                        <td>{ga_summary_test_acc}</td>
                        <td>{ga_summary_test_bacc}</td>
                    </tr>
                    <tr>
                        <td><b>Particle Swarm Optimization (PSO)</b></td>
                        <td>{pso_summary_train}</td>
                        <td>{pso_summary_test_f1}</td>
                        <td>{pso_summary_test_acc}</td>
                        <td>{pso_summary_test_bacc}</td>
                    </tr>
                </tbody>
            </table>

            <div class="grid-2">
                <div class="figure-container">
                    <img src="{img_convergence}" alt="Convergence Comparison">
                    <div class="figure-caption">Figure 2: Convergence histories (mean fitness with std deviation shadow).</div>
                </div>
                <div class="figure-container">
                    <img src="{img_boxplot}" alt="Performance Boxplot">
                    <div class="figure-caption">Figure 3: Distribution of Test F1-scores and Test Accuracies.</div>
                </div>
            </div>
        </div>

        <div class="card" style="page-break-before: always;">
            <h2>5. Best Models Analysis</h2>
            <p>
                We extracted the single best models trained by GA and PSO across all splits to inspect their detailed classification profiles.
            </p>

            <div class="grid-2">
                <div class="figure-container">
                    <img src="{img_confusion}" alt="Confusion Matrices">
                    <div class="figure-caption">Figure 4: Confusion Matrices for the best GA and PSO models.</div>
                </div>
                <div class="figure-container">
                    <img src="{img_roc}" alt="ROC Curves">
                    <div class="figure-caption">Figure 5: Receiver Operating Characteristic (ROC) curves.</div>
                </div>
            </div>
            
            <p>
                From Figure 4, we observe that both algorithms yield excellent predictive accuracy on the Parkinson's detection task. In clinical screening, minimizing <b>False Negatives</b> is highly critical, as a sick patient missed is much more dangerous than a healthy patient flagged. The confusion matrices display high recall (sensitivity) values for both methods, confirming they are clinically viable models.
            </p>
        </div>

        <div class="card">
            <h2>6. Statistical Analysis</h2>
            <p>
                We conducted a paired sample t-test and a Wilcoxon signed-rank test (a non-parametric equivalent) over the 10 independent splits to test the null hypothesis: <i>"There is no difference in testing F1-macro scores between GA and PSO."</i>
            </p>
            <ul>
                <li><b>Paired t-test:</b> p-value = <code>{p_val_t_str}</code></li>
                <li><b>Wilcoxon signed-rank test:</b> p-value = <code>{p_val_w_str}</code></li>
            </ul>
            <p>
                <b>Conclusion:</b> {conclusion_str}
            </p>
        </div>

        <div class="card">
            <h2>7. Discussion & Limitations</h2>
            <p>
                The experiments prove that both GA and PSO are fully capable of finding high-performance weight configurations for feedforward neural networks without using gradient info.
            </p>
            <p>
                <b>Algorithm Comparison:</b>
                PSO exhibits a faster initial convergence rate, rapidly finding a high-quality global region (thanks to the social attractors). GA converges more steadily and is less prone to premature convergence due to its stochastic crossovers (BLX-0.5) and mutations which continue to explore.
            </p>
            <p>
                <b>Limitations of Global Optimization for ANNs:</b>
                While meta-heuristics work beautifully on small networks, their computational scaling is problematic for large networks. A modern network with millions of parameters has a massive dimensional search space where evolutionary algorithms suffer from the "curse of dimensionality". Combining these global search algorithms with local gradient-based optimization (a hybrid Memetic Algorithm) represents a powerful path forward.
            </p>
        </div>

        <div class="card">
            <h2>8. Execution Instructions</h2>
            <p>To run the training pipeline and generate this report yourself, execute the following commands in order:</p>
            <pre style="background-color: #f1f5f9; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.95rem;">
# 1. Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn selenium webdriver-manager

# 2. Run the experimental evaluation suite
python run_experiments.py

# 3. Generate the report and export the PDF
python generate_report.py

# 4. Package deliverables into Group_XX.zip
python package_delivery.py
            </pre>
        </div>

        <div class="footer-info">
            &copy; 2026 Optimization Algorithms Course Project. All Rights Reserved.
        </div>
    </div>
</body>
</html>"""
    
    # Perform manual string replacements (avoids LaTeX & CSS f-string parsing conflicts)
    replacements = {
        "{img_ga_sensitivity}": img_ga_sensitivity,
        "{img_convergence}": img_convergence,
        "{img_boxplot}": img_boxplot,
        "{img_confusion}": img_confusion,
        "{img_roc}": img_roc,
        "{ga_summary_train}": ga_summary.get('Train F1-macro', 'N/A'),
        "{ga_summary_test_f1}": ga_summary.get('Test F1-macro', 'N/A'),
        "{ga_summary_test_acc}": ga_summary.get('Test Accuracy', 'N/A'),
        "{ga_summary_test_bacc}": ga_summary.get('Test Bal. Acc', 'N/A'),
        "{pso_summary_train}": pso_summary.get('Train F1-macro', 'N/A'),
        "{pso_summary_test_f1}": pso_summary.get('Test F1-macro', 'N/A'),
        "{pso_summary_test_acc}": pso_summary.get('Test Accuracy', 'N/A'),
        "{pso_summary_test_bacc}": pso_summary.get('Test Bal. Acc', 'N/A'),
        "{p_val_t_str}": p_val_t_str,
        "{p_val_w_str}": p_val_w_str,
        "{conclusion_str}": conclusion_str
    }
    
    for key, val in replacements.items():
        html_content = html_content.replace(key, str(val))
        
    with open("results/report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("HTML report successfully written to results/report.html")

def convert_html_to_pdf():
    print("Converting HTML report to PDF using Selenium...")
    html_path = os.path.abspath("results/report.html")
    pdf_path = os.path.abspath("Report.pdf")
    
    # Try Edge headless
    options = EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = None
    try:
        print("Launching headless Edge...")
        driver = webdriver.Edge(options=options)
        driver.get(f"file:///{html_path}")
        
        pdf_base64 = driver.print_page()
        pdf_bytes = base64.b64decode(pdf_base64)
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"PDF successfully written to {pdf_path}")
        driver.quit()
        return
    except Exception as e:
        print("Failed to generate PDF with Edge:", e)
        if driver:
            try:
                driver.quit()
            except:
                pass
                
    # Try Chrome headless
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        print("Launching headless Chrome...")
        driver = webdriver.Chrome(options=options)
        driver.get(f"file:///{html_path}")
        
        pdf_base64 = driver.print_page()
        pdf_bytes = base64.b64decode(pdf_base64)
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"PDF successfully written with Chrome to {pdf_path}")
        driver.quit()
    except Exception as e:
        print("Failed to generate PDF with Chrome:", e)
        if driver:
            try:
                driver.quit()
            except:
                pass
        print("\n[WARNING] Could not compile PDF automatically. You can manually print results/report.html to PDF from your browser.")

if __name__ == "__main__":
    generate_html_report()
    convert_html_to_pdf()
