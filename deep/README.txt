


nn_0
====
Variante eines neuronalen netzes. 
Diese Variante funktioniert. "NNdemo" fuehrt eine regression durch von zwei input variablen auf 
eine output variable f(a,b) = y. Wobei eigentlich eine Funktion f(a+b)=y vorgegeben ist. 

Input und Output sind leicht erweiterbar auf mehrere nodes, allerdings ist die visualisierung
momentan nur auf eine input variable und eine outputvariable ausgelegt. Wenn man auf 
die visualisierung verzichtet, dann kann jede beliebige Anzahl an nodes gewaehlt werden.

Die Visualisierung verwendet den "gnuplotWrapper" und was genau ausgegeben wird ist in
der Klasse "Settings" bzw. "RegressionSettings" (fuer das regressions-beispiel) 
gegeben. 

Problem mit dieser implementierung: Mir scheint die erweiterung mit dropout schwierig. 
Ausserdm ist dere trainings-algorithmus nicht gut vom neuronalen netz getrennt. 


Der trainings-algorithmus war einmal Scaled Gradient Descent. Aber er ist mittlerweile
veraendert und eigentlich besser als der "richtig" implementierte algorithmus. 
Aber da ist mit modernen algorithmen wahrscheinlich noch was rauszuholen. Ich vermute
dass sich dieser algorithmus aehnlich zu SGD (stochiastic gradient descent) verhaelt
was einer der guten modernen algorithmen ist. 



nn_1
====
Die neue variante eines neuronalen netzes. Dieses netz ist noch nicht funktionstuechtig. 
Die Ziele dieser implementierung sind folgende:

- Variablen die nur lokal benoetigt werden sollen nur lokal gehalten werden (beispielsweise
die delta-informationen bei der backpropagation. 

- Minimizer soll unabhaengig sein von der implementierung des neuronalen netzes.
Das heisst, dass der minimizer nur einen vektor von Werten vorgibt und dafuer
den zu minimierenden Wert (entspricht dem error des neuronalen netzes) und evtl. die
gradienten zurueckgibt (nur wenn der minimizer sie anfordert). 

- keine "Synapsis"-Klasse und vor allem keine "fixen" weights. Die besten weights
werden im neuronalen netz gespeichert, im prinzip kann aber jeglicher vektor von 
werten mit der richtigen groesse an das netz uebergeben werden und damit der Error
fuer diese weights berechnet werden. Das hat zwei Gruende: Einerseits muessen die 
weights die er minimizer berechnet nicht bei jedem berechnungsschritt in das neuronale
netz kopiert werden. Andererseits kann dann spaeter einmal das netz leichter auf 
multithreading erweitert werden, da das netz in jedem thread nur mit lokalen 
variablen arbeiten wuerde und erst beim zusammenfuegen der threads die werte
zusammengefuegt werden wuerden. 

- "synapsen" sind durch gleich laufende vektoren von iteratoren auf nodes bestimmt.
Ich will beliebige verbindungen bauen koennen. Also auch netze die nicht vollstaendig
verbunden sind. 

- moeglichkeit der erweiterung mit dropout (und evtl. maxout). Soll einfach
moeglich sein indem ein node-maskierungs-vektor vorgegeben wird der dann durch die
funktionen durchgezogen wird. 

- einbau von un-supervised learning soll moeglich sein. 


gnuplotWrapper
==============
Eine kleine wrapper-klasse um gnuplot. Habe ich im Internet gefunden und erweitert. 
Sie ermoeglicht recht einfach gnuplot-plots aus C++ heraus zu bauen und anzuzeigen. 



mnist
=====
Eine abwandlung eines codes den ich im internet gefunden habe zum lesen der MNIST
(handschrift) daten. 


lbfgs
=====
Ein versuch einer implementierung von LBFGS (low memory BFGS). (in abwandlung eines
python codes den ich gefunden habe). Weiss nicht ob ich das richtig implementiert
habe bzw. wie gut LBFGS laeuft.



readpng
=======
Wieder eine abwandlung von einem code den ich gefunden habe der libpng-turbo anspricht 
um mit ihr PNG bilder zu laden. 




unsupervised learning:
......................

So wie ich es verstanden habe laeuft das so ab: nehmen wir an, wir wollen ein netz mit 
3 hidden layern trainieren { I, H0, H1, H2, O}. (input, 0, 1, 2, Output) 

1. schritt
netzaufbau: I, H0, I (autoencoder) --> training von hidden layer H0

2. Schritt
netzaufbau: I, H0, H1, H0 --> die werte von H0 werden einfach berechnet. 
Dann werden die Werte H0 als input fuer das training von hidden layer H1 
sowohl als input als auch als output verwendet (autoencoder). Also
H0, H1, H0. 

3. Schritt
netzaufbau: I, H0, H1, H2, O --> die werte von H1 werden einfach berechnet 
(nur forward). Dann werden die Werte H1 als input und der gewuenschte output O 
fuer das training von hidden layer H2 verwendet. Ab hier ist das training
supervised. 

4. Schritt
netzaufbau: I, H0, H1, H2, O. --> Jetzt erfolgt noch (von den bisherigen
weights ausgehend) ein volles training des ganzen netzes. 
