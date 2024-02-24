# PD-Project

This repository is the implementation of Assessing Gait Dysfunction Severity in Parkinson’s Disease Using 2-Stream Spatial-Temporal Neural Network by Andrew Liang.

Parkinson’s disease (PD), a neurodegenerative disorder, significantly impacts the quality of life for millions of people worldwide. PD primarily impacts dopaminergic neurons in the brain’s substantia nigra, resulting in dopamine deficiency and gait impairments such as bradykinesia and rigidity. Currently, several well-established tools, such as the Movement Disorder Society-Unified Parkinson’s Disease Rating Scale (MDS-UPDRS) and Hoehn and Yahr (H&Y) Scale, are used for evaluating gait dysfunction in PD. While insightful, these methods are subjective, time-consuming, and often ineffective in early-stage diagnosis. Other methods using specialized sensors and equipment to measure movement disorders are cumbersome and expensive, limiting their accessibility. This study introduces a hierarchical approach to evaluate gait dysfunction in PD through video footage. The novel 2-stream spatial-temporal neural network (2S-STNN) leverages the spatial-temporal features from the two streams of skeletons and silhouettes for PD classification. This approach achieves an accuracy rate of 89.87% and outperforms other models in existing literature. Additionally, it correctly identifies PD in all patients who have the disease. Then, the study employs saliency maps to highlight critical body regions that significantly contribute to the network’s predictions. A statistically significant correlation between saliency values and PD categories demonstrates the effectiveness of the model decisions. Looking more closely, the study investigates 21 specific gait attributes to gain a more detailed quantification of gait disorders. Parameters such as walking pace, step length, and neck forward angle are identified as the most distinctive markers in differentiating healthy individuals and those with PD. This approach offers an accurate, explainable, quantitative, and accessible solution to evaluate gait impairment severity in PD.

This framework includes

1. [Extract 3D skeleton from videos via VIBE](https://github.com/mkocabas/VIBE)
2. [Extract silhouettes from videos via All-in-One-Gait](https://github.com/jdyjjj/All-in-One-Gait)
3. Form long-term gait energy images (LT-GEI) from silhouettes.
4. Extract spatial-temporal features from 3D skeleton via DD-Net.
5. Extract spatial-temporal features from LT-GEI via VGG16.
6. Predict PD categories by merging features from skeleton and silhouettes.
7. Create saliency maps to highlight the key body regions that influence model decisions
8. Create the gait features to provide continuous measurement

