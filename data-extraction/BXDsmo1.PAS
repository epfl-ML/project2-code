PROGRAM NameXYZ;

USES
  strings, sysutils;
TYPE
  epoch = RECORD
            state          : char;
            bin            : array[0..400] of single;
            EEGv,EMGv,temp : single;
          END;
VAR
  OUTf                     : text;
  EEGf                     : file of epoch;
  spec4                    : epoch;
  t,vs                     : integer;
  DIRstr                   : string;


BEGIN
  DIRstr:='./';


  assign(EEGf,DIRstr+'BDF08.smo'); reset(EEGf);
  assign(OUTf,DIRstr+'BDF08.out'); rewrite(OUTf);

 if FileExists(DIRstr + paramStr(1) + '.raw') Then
		Writeln('This fucking file exists.');

  FOR t:=1 TO 100 DO
  BEGIN
    read(EEGf,spec4);
    Writeln(spec4.state);

    CASE spec4.state OF
      'w','1','4': vs:=3;
      'n','2','5': vs:=2;
      'r','3','6': vs:=1;
      ELSE vs:=0;
    END;
    writeln(OUTf, t:10, vs:2, ' ', spec4.EEGv, ' ', spec4.EMGv);
  END;
  close(OUTf);
  close(EEGf);
END.