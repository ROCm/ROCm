.. meta::
   :description: How to fine-tune LLMs with ROCm
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, GPUs, Llama, accelerators

*******************************************
Use ROCm for fine-tuning LLMs
*******************************************

Fine-tuning is an essential technique in machine learning, where a pre-trained model, typically trained on a large-scale dataset, is further refined to achieve better performance and adapt to a particular task or dataset of interest.

With AMD GPUs, the fine-tuning process benefits from the parallel processing capabilities and efficient resource management, ultimately leading to improved performance and faster model adaptation to the target domain.

The ROCm™ software platform helps you optimize this fine-tuning process by supporting various optimization techniques tailored for AMD GPUs. It empowers the fine-tuning of large language models, making them accessible and efficient for specialized tasks. ROCm supports the broader AI ecosystem to ensure seamless integration with open frameworks, models, and tools. 

Throughout the following topics, this guide discusses the goals and :ref:`challenges of fine-tuning a large language
model <fine-tuning-llms-concept-challenge>` like Llama 2. In the
sections that follow, you'll find practical guides on libraries and tools to accelerate your fine-tuning.

For AI fine-tuning tutorials in Jupyter Notebook format, see `Tutorials for AI developers <https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/>`_.

- :doc:`Conceptual overview of fine-tuning LLMs <overview>`

- :doc:`Fine-tuning and inference <fine-tuning-and-inference>` using a
  :doc:`single-accelerator <single-gpu-fine-tuning-and-inference>` or
  :doc:`multi-accelerator <multi-gpu-fine-tuning-and-inference>` system.


