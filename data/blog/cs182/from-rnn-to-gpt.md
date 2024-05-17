---
title: rnn부터 gpt까지 이해해보기
date: 2024-05-12
tags:
  - RNN
  - LSTM
  - attention
  - machanism
  - transformer
  - architecture
  - gpt
  - bert
  - sequential
  - model
  - cs182
draft: true
summary: gpt를 이해하기 위해 필요한 것들. RNN 구조부터 GPT까지 차근차근 이해해보기
type: Blog
images:
---
# GPT 구조에 대해 빠르게 이해하기 위한 방법
open ai의 chat gpt가 발표되기 이전에도 transformer가 가진 강력함에 대해서는 많은 이야기를 들었었다. 당시에도 주변 사람들에게 물어보며 핵심적인 구조인 transformer와 self-attention에 대해 이해해보려고 했지만 성공하지는 못했었다. 지금 생각해보면 이해하기에는 내가 가진 지식이 너무 부족했던 것 같다. 결국에 gpt를 빨리 이해하기 위해서는 필요한 사전지식을 모두 갖추고 있어야 할 것이다. [cs182](https://www.youtube.com/watch?v=rSY1pVGdZ4I&list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A) 강의의 10장부터 13장까지가 RNN부터 GPT까지를 모두 다루고 있어 큰 도움이 되었다. 이글은 해당 강의들에 대한 요약 정리한 것이다. 강의 내용을 다시 되새기는 것에 목적이 있다. cs182 강의는 딥러닝을 처음 접한 사람보다는 조금 익숙한 사람들이 다시 정리하기 좋은 강의라고 생각된다.

# 전반적인 강의 흐름
아래 그림은 개략적인 흐름을 나타낸 것이다. cs182 강의의 장점은 스토리텔링에 있다고 생각한다. 우선은 RNN 계열부터 Transformer까지 모델의 구조적인 부분에 대한 부분을 다룬다. 그 이후에는 구조들을 기반으로 하여 방대한 양의 무가치해 보이는 데이터를 활용하는 방법에 대한 설명을 진행한다.
각 모델 별로 어떤 문제를 해결하기 위한 것인지 어떤 한계점이 있는 것인지 설명되어서 이해하기 좋은 것 같다. 예를 들면 naive RNN은 gradient vanishing이나 exploding 현상이 발생하기 때문에 이러한 한계를 보완하기 위해 LSTM 구조를 사용하게 되었다. 
![[summary-from-rnn-to-gpt.png]]
# RNN 계열의 알고리즘
## RNN 구조
우선 RNN 구조는 왜 필요한 것일까? 실생활에서는 입력 변수의 길이가 서로 다른 데이터가 더 일반적인 것 같다. 인터넷에 작성된 글이나 동영상, 음악 등 대부분 길이가 고정되어 있지 않다. 이러한 데이터를 활용하기 위해서는 어떻게 해야될까? 인풋 길이와 상관없이 적용가능한 구조를 만들어야할 것이다.
RNN은 개념적으로 입력 데이터 길이에 맞춰서 layer를 추가해주는 구조이다. 이렇게 하면 발생되는 문제점이 layer의 개수가 증가하며 이에 따라 학습이 필요한 weight와 bias 또한 같이 증가하게 된다. 이런 문제는 추가되는 layer의 가중치(weight)와 편향(bias)를 공유하도록 하여 해결하였다. 공유된 weight와 bias를 이용하면 입력의 길이와 상관없이 사용할 수 있게 된다. 마찬가지로 output에 대해서도 길이를 원하는 만큼 조절할 수 있게 된다.
## LSTM 구조
RNN 구조를 통해 입력과 출력 데이터의 길이를 원하는 만큼 조절하여 사용할 수 있게 되었다. 기울기가 폭주(exploding)나 소실(vanishing) 되는 현상이 발생된다는 한계가 여전히 존재한다. 기울기 폭주(gradient exploding) 대해서는 gradient clipping 같은 방법을 사용해서 해결이 가능하다고 한다. 그러나 기울기 소실은 쉽게 해결하기 어려워 이에 대한 대책으로 구조의 개선이 필요하다. 
![[lstm-cell-structure.png]]
위의 그림과 같이 구성하게 된다. 이 그림은 정말 많이 봐온 그림이긴 하지만 제대로 이해하기 어려웠다. 각각의 게이트가 가지는 의미를 알고나게 되니, 조금 더 이해가 쉬운 것 같다. $a_t$는 cell state 또는 long-term memory라고 불리며 $h_t$는 hidden state 또는 short-term memory라고 부른다. hidden state는 매 스텝마다 cell state와 곱해져서 계산 되기 때문에 비교적 빠르게 업데이트 된다. cell state의 경우에는 $f_t$,  $i_t$, $g_t$ 등 복잡한 계산을 거친다. $t_t$의 경우에는 이전 단계의 hidden state 그리고 현재의 input $x_t$가 곱해져 과거 상태를 얼마나 기억할 지에 대해서 결정한다. 그리고 $i_t$는 현재 상태를 기억할 지에 대한 결정, $g_t$는 얼마나 반영 시킬 지를 각각 결정하게 된다. 이런 방식으로 cell state는 천천히 변경되기 때문에 long-term memory라고 불리게 된다.
LSTM은 naive RNN에 비해서 매우 월등하게 복잡한 구조이지만, 성능이 우수하다. GRU의 경우에는 LSTM 유사한 아이디어를 가지고, 가볍게 작성된 모델이다. GRU가 대체로 연산 속도가 LSTM에 비해서 우수하고 성능은 유사하거나 다소 떨어진다. 일반적으로 RNN이라고 하면 naive RNN을 의미하는 경우보다는 LSTM이나 GRU를 의미하는 것으로 보인다 그래서 naive RNN라고 용어를 구별해서 사용하는 듯 하다.

아래 링크에 블로그에서 보다 상세하게 잘 작성되어있다.
[LSTM, GRU 이론 및 개념 참조 링크](https://blog.naver.com/winddori2002/221992543837)

## Attention 구조
Cell 구조를 개선하여 긴 step의 input에 대한 처리가 가능하도록 하였음에도, 구조가 가진 한계점이 여전히 존재한다. Sequence-to-sequence 모델의 경우 encoder와 decoder로 구성하게 된다. decoder는 encoder의 마지막 state에만 의존하게 된다. encoder의 step이 길어질 경우 처음 step에 대한 영향도는 낮아질 것이다. 이로 인해 장기의존성(long term dependencies) 문제가 생길 수 있다. 


attention과 관련해서는 아래 블로그 내용이 이해가 쉬웠다.
https://shyu0522.tistory.com/12



## Transformer 구조
Transformer를 이루는 네 가지 구조.


## 전체적인 요약

# Pre-trained model
## Representation Learning






