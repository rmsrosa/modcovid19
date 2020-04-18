# Modelagem Covid-19 - IM/UFRJ - Período 2020/1

[![Text License: CC-BY-NC-ND license](https://img.shields.io/badge/Text%20License-CC--BY--NC--ND-yellow.svg)](https://opensource.org/licenses/MIT) [![Code License: GNU-GPLv3](https://img.shields.io/badge/Code%20License-GNU--GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl.html) ![GitHub repo size](https://img.shields.io/github/repo-size/rmsrosa/nbbinder)

Notas sobre modelagem matemática da epidemia de Covid-19

[Ricardo M. S. Rosa](http://www.im.ufrj.br/rrosa/).

## *Links* de acesso direto aos cadernos

*Links* para acessar a página inicial das notas via [Github](https://github.com), [Binder](https://beta.mybinder.org/) e [Google Colab](http://colab.research.google.com):

[![Github](https://img.shields.io/badge/view%20on-github-orange)](contents/notebooks/00.00-Pagina_Inicial.ipynb) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rmsrosa/modcovid19/master?filepath=contents%2Fnotebooks%2F00.00-Pagina_Inicial.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rmsrosa/modcovid19/blob/master/contents/notebooks/00.00-Pagina_Inicial.ipynb)

## Observações

- As **notas de aula** estão dispostas na forma de uma coleção de [cadernos Jupyter](https://jupyter.org/) e estão disponíveis no subdiretório [contentes/notebooks](contents/notebooks). As notas serão escritas e disponibilizadas ao longo do curso/projeto.

- Há uma [Página Inicial](contents/notebooks/00.00-Pagina_Inicial.ipynb) exibindo a coleção de cadernos de forma estruturada, com links para cada caderno.

- Os cadernos podem ser visualizados dentro do próprio [Github](https://github.com), ou acessadas, modificadas e executadas nas "nuvens de computação" [Binder](https://beta.mybinder.org/) e [Google Colab](http://colab.research.google.com), através dos links exibidos acima.

- Além disso, um servidor [Jupyter Hub](https://jupyter.org/hub) próprio para o curso está disponível em [Jupyter Hub ModCovid19](https://www.modcovid19.ricardomsrosa.org/jupyter),  onde cada aluno terá uma conta e todo o ambiente computacional necessário para a execução dos notebooks.

- O servidor [Jupyter Hub ModCovid19](https://www.modcovid19.ricardomsrosa.org/jupyter) foi instalado em uma instância da [Amazon EC2 (Amazon Elastic Compute Cloud)](https://aws.amazon.com/pt/ec2/), do tipo [t2.micro](https://aws.amazon.com/pt/ec2/instance-types/), com apenas 1 vCPU, 1Gb RAM, 8Gb SSD. Não é muito, mas serve para uma primeira experiência e serve para a execução dos trabalhos a serem entregues via [nbgrader](https://nbgrader.readthedocs.io/). Mas, para uso geral, recomendamos uma máquina local ou uma das outras nuvens de computação ([Binder](https://beta.mybinder.org/), [Google Colab](http://colab.research.google.com), [Kaggle](https://www.kaggle.com/), por exemplo).

- Cada aluno deverá enviar, para o email do professor, o seu **nome**, **sobrenome**, **email** e o **nome de usuário** que deseja ter no [Jupyter Hub ModCovid19](https://www.modcovid19.ricardomsrosa.org/jupyter).

- Ao longo do período, é esperado que os alunos modifiquem os cadernos existentes e criem os seus próprios cadernos para resolver os exercícios, os testes e escrever os trabalhos/mini-projetos e o projeto final.

- A comunicação mais apropriada entre o professor e os alunos ainda está sendo avaliada, mas uma possibilidade é que isso seja feito através do [AVA @ UFRJ (Ambiente Virtual de Aprendizagem na UFRJ)](http://ambientevirtual.nce.ufrj.br/). Informes via [SIGA/Intranet UFRJ](https://intranet.ufrj.br/) e mensagens diretas via *e-mail* também podem ser utilizados.

- Todo o conteúdo do repositório pode ser baixado para uma máquina local através do botão `Clone or Download` da página inicial [rmsrosa/modcovid19](https://github.com/rmsrosa/modcovid19) e escolhendo a opção `Download ZIP`.

- Cada caderno, além de poder ser visualizado diretamente no [Github](https://github.com) e acessado nas nuvens de computação, também pode ser baixado individualmente para uma máquina local clicando-se no ícone `Raw`, que aparece em cada página, e baixando para a sua máquina o conteúdo que aparecer no navegador (é um arquivo fonte de cadernos [jupyter](https://jupyter.org/), com a extensão `".ipynb"`).

- As alterações nos cadernos deste repositório e a criação de novos cadernos podem ser feitas localmente, em máquinas com o Python (versão 3.8) e os devidos pacotes devidamente instalados, ou nas nuvens de computação mencionadas acima.

- A lista dos pacotes python necessários para a execução do conjunto de cadernos aparece no arquivo [requirements.txt](requirements.txt). Esse arquivo não é apenas uma referência, ele é necessário para o [Binder](https://beta.mybinder.org/) poder montar o ambiente python com todos os pacotes a serem utilizados. O [Google Colab](http://colab.research.google.com), por outro lado, já tem o seu próprio ambiente, bastante completo, e não depende deste arquivo. O ambiente no [Jupyter Hub ModCovid19](https://www.modcovid19.ricardomsrosa.org/jupyter), por sua vez, já conta com os pacotes devidamente pré-instalados.

- No [Binder](https://beta.mybinder.org/) e no [Google Colab](http://colab.research.google.com), um ambiente python temporário é montado e os cadernos podem ser alterados e executados interativamente. Mas eles não são guardados para uma próxima sessão. Se quiser salvar as alterações, é necessário baixar os cadernos alterados para a sua máquina.

- Uma alternativa, caso tenha o [Google Drive](https://www.google.com/drive/), é habilitar o [Google Colab](http://colab.research.google.com) em sua conta do Google e copiar as notas para um diretório denominado *Colab Notebooks* que será automaticamente criado em seu [Google Drive](https://www.google.com/drive/). Nesse caso, as notas podem ser acessadas, executadas e gravadas normalmente para uso posterior, como se estivesse com uma instalação local do [jupyter](https://jupyter.org/).

- Vale ressaltar, no entanto, que o funcionamento do jupyter no [Google Colab](http://colab.research.google.com) é um pouco diferente do padrão e o acesso aos arquivos locais é um pouco mais delicado. Por conta disso, alguns notebooks poderão não funcionar sem pequenas modificações de acesso aos arquivos.

- Uma outra alternativa é criar uma conta no [github](https://github.com), *clonar* o repositório e usar o [Google Colab](http://colab.research.google.com) ou o [Binder](https://beta.mybinder.org/) a partir do seu repositório. Será necessário, no entanto, após a clonagem, modificar os cadernos para atualizar os links com o nome do seu repositório. Trabalhar com o github não é trivial, mas uma vantagem é que será mais fácil submeter correções ou sugestões para este repositório, ajudando-o a melhorar, assim como receber qualquer atualização de maneira mais suave.

- Abrir um conta no [github](https://github.com) também permite marcar este repositório com uma "estrela", para acesso direto a partir do seu perfil. É uma espécie de "bookmark". Isso pode ser feito clicando-se no botão *Star*, no canto superior direito do repositório.

- Outra opção é clicar no botão *Watch*, também no canto superior direito do repositório. Dessa forma, você receberá notificações, por *e-mail*, sobre qualquer modificação feita no mesmo.

## Licença

Os **textos** neste repositório estão disponíveis sob a licença [CC-BY-NC-ND license](LICENSE-TEXT). Mais informações sobre essa licença em [Creative Commons Attribution-NonCommercial-NoDerivs 3.0](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode).

Os **códigos** neste repositório, nos blocos de código dos [jupyter notebooks,](https://jupyter.org/) estão disponíveis sob a [licença GNU-GPL](LICENSE-CODE). Mais informações sobre essa licença em [GNU GENERAL PUBLIC LICENSE Version 3](https://www.gnu.org/licenses/gpl.html).
