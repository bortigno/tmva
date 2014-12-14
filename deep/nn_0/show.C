{
    TTree train;
    train.ReadFile ("train.csv");
    TTree test;
    test.ReadFile ("test.csv");
    
    new TCanvas ("train","train");
    train.Draw ("t:s","","colz");
    train.Draw ("o:s","","colz same");

    new TCanvas ("test","test");
    train.Draw ("t:s","","colz");
    train.Draw ("o:s","","colz same");
}
