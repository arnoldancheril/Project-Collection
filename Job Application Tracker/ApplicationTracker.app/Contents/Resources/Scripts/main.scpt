FasdUAS 1.101.10   ��   ��    k             l     ��  ��    / ) Application Tracker Launcher AppleScript     � 	 	 R   A p p l i c a t i o n   T r a c k e r   L a u n c h e r   A p p l e S c r i p t   
  
 l     ��  ��    > 8 This script creates a proper macOS application launcher     �   p   T h i s   s c r i p t   c r e a t e s   a   p r o p e r   m a c O S   a p p l i c a t i o n   l a u n c h e r      l     ��������  ��  ��     ��  i         I     ������
�� .aevtoappnull  �   � ****��  ��    Q     g     k    :       l   ��  ��    ? 9 Set the correct path to your Application Tracker project     �   r   S e t   t h e   c o r r e c t   p a t h   t o   y o u r   A p p l i c a t i o n   T r a c k e r   p r o j e c t      r         m     ! ! � " " ^ / U s e r s / a r n o l d a n c h e r i l / D e s k t o p / a p p t r a c k e r . n o s y n c   o      ����  0 apptrackerpath appTrackerPath   # $ # l   ��������  ��  ��   $  % & % l   �� ' (��   ' , & Launch the application using Terminal    ( � ) ) L   L a u n c h   t h e   a p p l i c a t i o n   u s i n g   T e r m i n a l &  * + * O    8 , - , k    7 . .  / 0 / l   �� 1 2��   1 + % Check if Terminal is already running    2 � 3 3 J   C h e c k   i f   T e r m i n a l   i s   a l r e a d y   r u n n i n g 0  4 5 4 Z    1 6 7�� 8 6 H     9 9 l    :���� : I   �� ;��
�� .coredoexnull���     obj  ; 4    �� <
�� 
cwin < m    ���� ��  ��  ��   7 k      = =  > ? > l   �� @ A��   @ ' ! Open a new window if none exists    A � B B B   O p e n   a   n e w   w i n d o w   i f   n o n e   e x i s t s ?  C�� C I    �� D��
�� .coredoscnull��� ��� ctxt D b     E F E b     G H G m     I I � J J  c d   ' H o    ����  0 apptrackerpath appTrackerPath F m     K K � L L 8 '   & &   . / l a u n c h _ a p p _ t r a c k e r . s h��  ��  ��   8 k   # 1 M M  N O N l  # #�� P Q��   P   Use existing window    Q � R R (   U s e   e x i s t i n g   w i n d o w O  S�� S I  # 1�� T U
�� .coredoscnull��� ��� ctxt T b   # ( V W V b   # & X Y X m   # $ Z Z � [ [  c d   ' Y o   $ %����  0 apptrackerpath appTrackerPath W m   & ' \ \ � ] ] 8 '   & &   . / l a u n c h _ a p p _ t r a c k e r . s h U �� ^��
�� 
kfil ^ 4   ) -�� _
�� 
cwin _ m   + ,���� ��  ��   5  ` a ` l  2 2��������  ��  ��   a  b c b l  2 2�� d e��   d   Bring Terminal to front    e � f f 0   B r i n g   T e r m i n a l   t o   f r o n t c  g�� g I  2 7������
�� .miscactvnull��� ��� null��  ��  ��   - m     h h�                                                                                      @ alis    J  Macintosh HD               �<K�BD ����Terminal.app                                                   �����<K�        ����  
 cu             	Utilities   -/:System:Applications:Utilities:Terminal.app/     T e r m i n a l . a p p    M a c i n t o s h   H D  *System/Applications/Utilities/Terminal.app  / ��   +  i�� i l  9 9��������  ��  ��  ��    R      �� j��
�� .ascrerr ****      � **** j o      ���� 0 errmsg errMsg��    k   B g k k  l m l l  B B�� n o��   n 0 * Show error dialog if something goes wrong    o � p p T   S h o w   e r r o r   d i a l o g   i f   s o m e t h i n g   g o e s   w r o n g m  q�� q I  B g�� r s
�� .sysodlogaskr        TEXT r b   B M t u t b   B I v w v b   B G x y x b   B E z { z m   B C | | � } } J E r r o r   l a u n c h i n g   A p p l i c a t i o n   T r a c k e r :   { o   C D���� 0 errmsg errMsg y o   E F��
�� 
ret  w o   G H��
�� 
ret  u m   I L ~ ~ �   � M a k e   s u r e   t h e   a p p   i s   l o c a t e d   a t :   / U s e r s / a r n o l d a n c h e r i l / D e s k t o p / a p p t r a c k e r . n o s y n c s �� � �
�� 
btns � J   P U � �  ��� � m   P S � � � � �  O K��   � �� � �
�� 
dflt � m   X [ � � � � �  O K � �� ���
�� 
disp � m   ^ a��
�� stic    ��  ��  ��       �� � ���   � ��
�� .aevtoappnull  �   � **** � �� ���� � ���
�� .aevtoappnull  �   � ****��  ��   � ���� 0 errmsg errMsg �  !�� h���� I K�� Z \�������� |�� ~�� ��� �����������  0 apptrackerpath appTrackerPath
�� 
cwin
�� .coredoexnull���     obj 
�� .coredoscnull��� ��� ctxt
�� 
kfil
�� .miscactvnull��� ��� null�� 0 errmsg errMsg��  
�� 
ret 
�� 
btns
�� 
dflt
�� 
disp
�� stic    �� 
�� .sysodlogaskr        TEXT�� h <�E�O� .*�k/j  ��%�%j Y ��%�%�*�k/l O*j UOPW ,X  �%�%�%a %a a kva a a a a  ascr  ��ޭ